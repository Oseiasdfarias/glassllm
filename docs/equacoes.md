# As equações de um LLM, com as deduções

Este texto acompanha o código do `glassllm`. Ele segue o caminho de um token pelo modelo e,
em cada passo, mostra a equação, de onde ela vem e por que ela tem aquela forma.

Notação: $B$ é o tamanho do batch, $n$ o número de tokens novos da entrada,
$m$ o número total de posições K/V, $d$ o tamanho do vetor de cada token,
$V$ o tamanho do vocabulário, $L$ o número de blocos, $H$ o número de cabeças Q,
$H_{kv}$ o número de cabeças K/V e $d_k = d/H$ o tamanho de cada cabeça.
Sem cache, $m=n$; com cache, $m=\text{start}+n$.

A configuração padrão do código NumPy é $V=1024$, $d=128$, $L=4$, $H=4$,
$H_{kv}=2$, $d_k=32$, contexto máximo de 128 posições e largura FFN de 341.
Usamos vetores linha, como nas multiplicações `x @ W` do código.

O protótipo original, em PyTorch, está em [prototipo/mini_llm_torch.py](prototipo/mini_llm_torch.py).

---

## 0. A ideia que organiza tudo

Um LLM dá uma probabilidade para um texto. A regra da cadeia da probabilidade quebra essa conta
em pedaços:

$$
P(x_1, \dots, x_n) = P(x_1)\prod_{t=1}^{n-1} P(x_{t+1} \mid x_1, \dots, x_t)
$$

Basta saber responder "dado o que veio antes, qual o próximo token?". O Transformer é uma
função que recebe os tokens anteriores e devolve uma distribuição sobre o vocabulário.
Tudo o que vem abaixo serve a essa função.
Na geração da biblioteca, o prompt fornece o prefixo: o motor prevê sua continuação,
sem inserir automaticamente um token BOS.

## 1. Embedding: transformar ID em vetor

Um token é um número inteiro, e esse número não tem geometria. O primeiro passo dá a cada
token um ponto num espaço de $d$ dimensões.

Escreva o token $i$ como um vetor one-hot $e_i \in \mathbb{R}^V$. Multiplicar pela matriz de
embedding $E \in \mathbb{R}^{V \times d}$ dá:

$$
x = e_i^\top E = \text{linha } i \text{ de } E
$$

O embedding é uma camada linear aplicada a um one-hot. Na prática se pega a linha $i$
direto (lookup). As linhas começam aleatórias e são ajustadas pelos gradientes do treinamento;
coordenadas ou distâncias específicas não têm uma interpretação linguística garantida.

## 2. Atenção

**O problema.** Depois do embedding, cada vetor só sabe sobre o próprio token. Cada posição
precisa juntar informação das anteriores.

**Média ponderada.** A forma mais geral de juntar é
$y_i = \sum_j \alpha_{ij} x_j$, com $\alpha_{ij} \ge 0$ e $\sum_j \alpha_{ij} = 1$.
Falta decidir os pesos a partir do conteúdo.

**Parecença.** O produto escalar $s_{ij} = x_i \cdot x_j$ mede o quanto dois vetores combinam.
Para virar peso, ele precisa ficar positivo e somar 1. A exponencial resolve a primeira parte
e dividir pela soma resolve a segunda:

$$
\alpha_{ij} = \frac{e^{s_{ij}}}{\sum_k e^{s_{ik}}} \quad \text{(softmax)}
$$

Se todos os scores forem multiplicados por um número positivo cada vez maior, o softmax
concentra massa no maior score. Com um máximo único, o limite é um one-hot; em empate,
a massa divide-se entre os máximos. Por isso ele é um "argmax suave", diferenciável.

**Q, K e V.** Usar $x_i \cdot x_j$ direto é simétrico (i gosta de j tanto quanto j gosta de i)
e faz o mesmo vetor cumprir três papéis. Damos uma matriz aprendida para cada papel:

$$
q_i = x_i W_Q \;(\text{pergunta}), \qquad k_j = x_j W_K \;(\text{etiqueta}), \qquad v_j = x_j W_V \;(\text{conteúdo})
$$

É um dicionário suave: a pergunta combina um pouco com cada etiqueta, e o resultado é a mistura
dos conteúdos. Em forma de matriz:

$$
\text{Attention}(Q, K, V) = \text{softmax}(QK^\top) V
$$

**De onde vem o $\sqrt{d_k}$.** Suponha componentes de $q$ e $k$ independentes, com média 0 e
variância 1:

$$
q \cdot k = \sum_{l=1}^{d_k} q_l k_l, \qquad
\mathbb{E}[q_l k_l] = 0, \qquad
\text{Var}(q_l k_l) = \mathbb{E}[q_l^2]\,\mathbb{E}[k_l^2] = 1
\;\Rightarrow\; \text{Var}(q \cdot k) = d_k
$$

No modelo padrão, $d_k = 32$ e esse desvio-padrão, sob as hipóteses acima, é
$\sqrt{32}\approx 5{,}66$ antes da divisão. Não é uma medida dos scores aprendidos.
A derivada do softmax é

$$
\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j)
$$

e, quando $p$ é quase one-hot, as derivadas do softmax tornam-se pequenas.
Isso pode dificultar a propagação de gradientes pela atenção; não significa que todo gradiente
do modelo seja zero. Dividir por $\sqrt{d_k}$ volta a variância para 1 sob as hipóteses acima:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
$$

**Máscara causal.** A posição $t$ só pode usar $x_1, \dots, x_t$. Somamos

$$
M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

Como $e^{-\infty} = 0$, o futuro recebe peso zero. O ganho: numa única passada, cada posição
prevê o token seguinte, ou seja, $n$ previsões em paralelo. Treinar é paralelo; gerar é
sequencial.

Com cache e índices começando em zero, a posição da consulta local $i$ é
$\text{start}+i$: a máscara proíbe $j>\text{start}+i$, e tem dimensões $n\times m$.
`causal_mask(n, m, start)` retorna `True` nos pontos proibidos; `masked_fill` insere $-\infty$.
O softmax subtrai o máximo de cada linha antes de exponenciar, sem alterar as probabilidades.

## 3. Posição e RoPE

Sem uma representação explícita de posição, os produtos escalares Q/K dependem de seus
conteúdos, sem uma distância posicional codificada diretamente. A máscara causal já fornece
uma restrição de ordem; RoPE acrescenta posição às comparações Q/K.

**O que queremos:** para vetores Q/K fixos, a contribuição posicional ao score entre as
posições $p$ e $q$ deve depender da distância $p-q$.

**Dedução em 2D.** Trate um par de Q como o complexo $z_Q=Q_1+iQ_2$, e um par de K
como $z_K=K_1+iK_2$. Em 2D, $a\cdot b=\text{Re}(a\bar b)$.
Girando Q pelo ângulo $p\theta$ e K por $q\theta$:

$$
f(Q, p) \cdot f(K, q)
= \text{Re}\!\left(z_Q e^{ip\theta}\, \bar z_K e^{-iq\theta}\right)
= \text{Re}\!\left(z_Q \bar z_K\, e^{i(p-q)\theta}\right)
$$

Na parte posicional, só sobrou $p-q$. Em números reais, isso é a matriz de rotação
de um par de coordenadas da consulta na posição $p$:

$$
\begin{pmatrix} Q'_1 \\ Q'_2 \end{pmatrix} =
\begin{pmatrix} \cos p\theta & -\sin p\theta \\ \sin p\theta & \cos p\theta \end{pmatrix}
\begin{pmatrix} Q_1 \\ Q_2 \end{pmatrix}
$$

**Em $d_k$ dimensões:** cada par $(2i, 2i+1)$ gira com a sua frequência

$$
\theta_i = 10000^{-2i/d_k}, \qquad i = 0, \dots, d_k/2 - 1
$$

Como os ponteiros de um relógio: pares rápidos distinguem posições próximas, pares lentos
distinguem posições distantes. O RoPE entra só em $Q$ e $K$, dentro de toda camada.

As tabelas de seno e cosseno têm shape `(128, 16)` na configuração padrão. Durante a
inferência incremental, o código usa as linhas `start:start+n`. K é armazenado depois da
rotação e não deve ser rotacionado novamente ao consultar o cache. A janela do motor é
limitada a 128 posições: RoPE não habilita extrapolação automática nessa implementação.

## 4. Multi-head e GQA

Cada linha do softmax produz uma única média ponderada. Para o token prestar atenção em várias
relações ao mesmo tempo, dividimos $d$ em $H$ cabeças:

$$
\text{head}_h = \text{Attention}(X W_Q^h, X W_K^h, X W_V^h), \qquad
\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)\, W_O
$$

Somar os parâmetros das projeções Q por cabeça dá $H\cdot(d\cdot d/H)=d^2$.
Na atenção multi-head convencional, Q, K, V e O somam $4d^2$ parâmetros, sem bias.
Concatenar e multiplicar por $W_O$ equivale a $\sum_h \text{head}_h W_O^h$:
cada cabeça escreve sua contribuição. A memória dos pesos de atenção também depende de H.

**GQA:** $Q$ tem $H$ cabeças, $K$ e $V$ têm só $H_{kv}$, e grupos de $H/H_{kv}$ cabeças de $Q$
dividem o mesmo $K$ e $V$. O KV cache encolhe pelo fator $H/H_{kv}$.

Com $H=4$ e $H_{kv}=2$, a correspondência é `[0, 0, 1, 1]`: as cabeças Q 0/1 usam
K/V 0, e Q 2/3 usam K/V 1. `repeat_kv` expande `(B,2,m,32)` para `(B,4,m,32)` com
`np.repeat`; essa expansão temporária é materializada, mas o cache conserva apenas as
duas cabeças K/V. O backward soma as contribuições das consultas que compartilharam K/V.

`GlassLM.generate` primeiro processa todo o prompt; nos passos seguintes, processa o token
novo e consulta as chaves anteriores. Cada camada tem seu próprio `KVCache`. Quando uma
inserção ultrapassaria 128 posições, o motor reconstrói o cache a partir dos últimos 128
tokens, reiniciando as posições em zero. Não descarta apenas a chave mais antiga: os estados
anteriores dependiam de um histórico que saiu da janela. Esse cache é para inferência;
a concatenação de arrays não preserva o grafo de gradientes do histórico.

## 5. Conexão residual

O bloco calcula $x + F(x)$. A derivada é

$$
\frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F}{\partial x_l}
$$

Na expansão do produto de Jacobianos $(I+J_l)$ existe um termo de identidade, que oferece
uma contribuição direta ao gradiente. Isso ajuda a otimização, mas não garante que o
gradiente nunca desapareça ou exploda: termos podem se cancelar, como em $F(x)=-x$,
ou amplificar-se no produto. Cada camada lê do "fluxo residual" e soma seu resultado de volta.

## 6. Normalização

Ao somar saídas no fluxo residual, a escala dos vetores pode variar.

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta
\qquad\qquad
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}}
$$

A RMSNorm remove a centralização e o parâmetro beta da expressão de LayerNorm.
No código, cada RMSNorm aprende um vetor gamma e usa $\varepsilon=10^{-6}$.
Para $c>0$:

$$
\text{RMSNorm}(c\,x)
=\gamma\odot\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\varepsilon/c^2}}.
$$

A invariância de escala é exata na expressão ideal com $\varepsilon=0$ para vetores não
nulos; com epsilon positivo, é aproximada quando a média dos quadrados domina epsilon.
Por exemplo, normalizar `[3,4]` e `[6,8]`, com gamma 1 e epsilon $10^{-6}$, produz diferenças
de aproximadamente `[2,55 × 10⁻⁸, 3,39 × 10⁻⁸]`. RMSNorm não impõe média zero, nem variância
um, e um gamma não uniforme também altera a escala por coordenada.

**Pre-norm.** O original fazia $\text{Norm}(x + F(x))$, o que põe a norma no caminho direto da
seção 5. O pre-norm deixa o caminho limpo:

$$
H = X + \text{Attention}(\text{RMSNorm}_1(X)), \qquad
X' = H + \text{SwiGLU}(\text{RMSNorm}_2(H))
$$

No projeto, Attention usa GQA. Há duas RMSNorms por bloco e uma RMSNorm final antes dos
logits. As duas somas residuais de cada bloco preservam shape `(B,n,128)`.

## 7. Feed-forward (FFN)

A atenção mistura informação entre tokens. O FFN processa cada token sozinho, com não
linearidade:

$$
\text{FFN}(x) = \varphi(x W_1) W_2, \qquad W_1: d \to 4d
$$

Uma leitura útil, sem garantir uma interpretação individual dos neurônios: cada coluna de
$W_1$ detecta uma combinação da entrada e cada linha de $W_2$ contribui para a saída.

**SwiGLU:**

$$
\text{FFN}(x) = \big(\text{SiLU}(x W_1) \odot x W_3\big) W_2,
\qquad \text{SiLU}(z) = z\,\sigma(z)
$$

Um caminho decide quanto passa, o outro leva o que passa. Com três matrizes, para manter
$8d^2$ parâmetros: $3 \cdot d \cdot h = 8d^2 \Rightarrow h = 8d/3$.

O código arredonda com `hidden = int(8*d/3)`: para $d=128$, $h=341$.
W1 e W3 têm shape `(128,341)`, W2 tem `(341,128)`, sem bias. Isso resulta em
130.944 parâmetros por FFN, ligeiramente abaixo de $8d^2$ pelo arredondamento.
O portão SiLU pode ser negativo e não é uma probabilidade limitada a `[0,1]`.
O snapshot `ffn.act` contém `SiLU(gate) * up`, não apenas SiLU.

## 8. Saída e temperatura

$$
z = h W_{\text{out}}, \qquad P(\text{próximo} = i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

Com a convenção de vetores linha, os pesos amarrados são $W_{\text{out}}=E^\top$:
o código calcula `logits = h @ E.T`, sem matriz de saída treinável independente e sem bias.
A saída do forward tem shape `(B,n,1024)`; a geração usa somente sua última posição.
A temperatura age na razão entre probabilidades:

$$
\frac{p_i}{p_j} = e^{(z_i - z_j)/T}
$$

Para logits finitos, $T\to0^+$ concentra massa nos máximos, dividindo-a em caso de empate,
e $T\to\infty$ tende à distribuição uniforme. `next_token_probs` e `generate` exigem
temperatura positiva. A escolha determinística usa `greedy=True`, que aplica argmax;
o modo padrão sorteia com `rng.choice`, opcionalmente com semente fixa.
`generate` não interrompe automaticamente ao escolher EOS: o limite é `n_new`.

## 9. Perda e o gradiente $p - y$

Maximizar $\prod_t P(x_{t+1} \mid x_{\le t})$ é o mesmo que minimizar

$$
\mathcal{L} = -\frac{1}{n} \sum_t \log P(x_{t+1} \mid x_1, \dots, x_t)
$$

que é a entropia cruzada entre o one-hot $y$ e a previsão $p$. Para a perda individual
$\ell$ de uma posição com alvo $c$:

$$
\ell = -z_c + \log \sum_j e^{z_j}
\qquad\Rightarrow\qquad
\frac{\partial \ell}{\partial z_i} = -\delta_{ic} + \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i - y_i
$$

Para a média sobre $N=B\cdot n$ posições, usada em `GlassLM.loss`, o gradiente é
$(p-y)/N$. Em uma atualização por descida de gradiente, o logit correto tende a subir e os
concorrentes a descer. `loss(ids, targets)` não desloca os alvos: `get_batch` fornece targets
uma posição à direita da entrada. A perda considera todas as posições, sem ignorar PAD.

Teste de sanidade: uma distribuição exatamente uniforme tem perda $\ln V$;
para $V=1024$, cerca de 6,931472 nats. Uma inicialização aleatória não é necessariamente
uniforme. `loss.grad` registra o gradiente analítico em relação aos logits, mas não executa
`Tensor.backward()` nem uma atualização dos parâmetros. `generate` usa `no_grad()`.

## 10. Contas para a arquitetura do projeto

- **Atenção GQA:** com $r=H_{kv}/H$, Q e O somam $2d^2$; K e V somam $2rd^2$.
  Portanto são $2d^2(1+r)=49.152$ parâmetros por bloco na configuração padrão.
- **FFN e normalizações:** a FFN tem $3dh=130.944$ parâmetros e as duas RMSNorms
  têm $2d=256$. Total por bloco: 180.352 parâmetros.
- **Modelo completo, com embedding compartilhado:**
  $P=Vd+L[2d^2(1+r)+3dh+2d]+d=852.608$.
  O último $d$ conta gamma da RMSNorm final; a saída não acrescenta outra matriz $Vd$.
- **Produtos de atenção:** para n consultas e m chaves, $O(nmd)$.
  Sem cache, $m=n$, e o custo é quadrático no contexto; na etapa incremental usual,
  $n=1$ e os produtos percorrem as m chaves. Isso exclui o custo das projeções.
- **FLOPs por token:** regras como $\approx2P$ na inferência e $\approx6P$ no treino
  são aproximações para operações densas dominantes, não medições deste motor.
  O tamanho do contexto, o lookup do embedding, os snapshots e os detalhes dos kernels
  impedem tratá-las como contagens exatas ou estimativas diretas de tempo.
- **K/V armazenados:**
  $2\cdot B\cdot L\cdot m\cdot H_{kv}\cdot d_k\cdot\text{bytes por elemento}$.
  Com batch 1 e 128 posições, são 65.536 elementos: 256 KiB em float32 ou 512 KiB em
  float64. Confira o dtype dos arrays e some `k.nbytes + v.nbytes` em todas as camadas:
  operações do autograd podem promover ativações a float64 mesmo com parâmetros float32.
  Essa conta exclui arrays temporários e snapshots; GQA reduz pela metade o cache
  comparado a quatro cabeças K/V do mesmo tamanho.
