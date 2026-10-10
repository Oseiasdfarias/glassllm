# As equações de um LLM, com as deduções

Este texto acompanha o código do `glassllm`. Ele segue o caminho de um token pelo modelo e,
em cada passo, mostra a equação, de onde ela vem e por que ela tem aquela forma.

Notação: $n$ é o número de tokens da entrada, $d$ o tamanho do vetor de cada token,
$V$ o tamanho do vocabulário, $L$ o número de blocos, $H$ o número de cabeças de atenção
e $d_k = d/H$ o tamanho de cada cabeça.

O protótipo original, em PyTorch, está em [prototipo/mini_llm_torch.py](prototipo/mini_llm_torch.py).

---

## 0. A ideia que organiza tudo

Um LLM dá uma probabilidade para um texto. A regra da cadeia da probabilidade quebra essa conta
em pedaços:

$$
P(x_1, \dots, x_n) = \prod_{t} P(x_{t+1} \mid x_1, \dots, x_t)
$$

Basta saber responder "dado o que veio antes, qual o próximo token?". O Transformer é uma
função que recebe os tokens anteriores e devolve uma distribuição sobre o vocabulário.
Tudo o que vem abaixo serve a essa função.

## 1. Embedding: transformar ID em vetor

Um token é um número inteiro, e esse número não tem geometria. O primeiro passo dá a cada
token um ponto num espaço de $d$ dimensões.

Escreva o token $i$ como um vetor one-hot $e_i \in \mathbb{R}^V$. Multiplicar pela matriz de
embedding $E \in \mathbb{R}^{V \times d}$ dá:

$$
x = e_i^\top E = \text{linha } i \text{ de } E
$$

O embedding é uma camada linear aplicada a um one-hot. Na prática se pega a linha $i$
direto (lookup). As linhas começam aleatórias, e o gradiente aproxima tokens de uso parecido.

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

Se todos os scores forem multiplicados por um número grande, o softmax tende a um one-hot no
maior score. Por isso ele é um "argmax suave": escolhe o mais parecido, mas deixa o gradiente
passar.

**Q, K e V.** Usar $x_i \cdot x_j$ direto é simétrico (i gosta de j tanto quanto j gosta de i)
e faz o mesmo vetor cumprir três papéis. Damos uma matriz aprendida para cada papel:

$$
q_i = W_Q x_i \;(\text{pergunta}), \qquad k_j = W_K x_j \;(\text{etiqueta}), \qquad v_j = W_V x_j \;(\text{conteúdo})
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

Com $d_k = 128$, os scores ficam na casa de $\pm 11$. A derivada do softmax é

$$
\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j)
$$

e, quando $p$ é quase one-hot, ela é quase zero em todo lugar: o gradiente some. Dividir por
$\sqrt{d_k}$ volta a variância para 1:

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

## 3. Posição e RoPE

Sem informação de posição, a atenção não distingue "cão morde homem" de "homem morde cão":
os pesos dependem só do conteúdo.

**O que queremos:** o score entre as posições $m$ e $n$ deve depender do conteúdo e só da
distância $m - n$.

**Dedução em 2D.** Trate o par $(q_1, q_2)$ como o complexo $z_q = q_1 + i q_2$. Em 2D,
$a \cdot b = \text{Re}(a \bar b)$. Girando $q$ pelo ângulo $m\theta$ e $k$ por $n\theta$:

$$
f(q, m) \cdot f(k, n)
= \text{Re}\!\left(z_q e^{im\theta}\, \bar z_k e^{-in\theta}\right)
= \text{Re}\!\left(z_q \bar z_k\, e^{i(m-n)\theta}\right)
$$

Só sobrou $m - n$. Em números reais, isso é a matriz de rotação:

$$
\begin{pmatrix} q'_1 \\ q'_2 \end{pmatrix} =
\begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}
\begin{pmatrix} q_1 \\ q_2 \end{pmatrix}
$$

**Em $d_k$ dimensões:** cada par $(2i, 2i+1)$ gira com a sua frequência

$$
\theta_i = 10000^{-2i/d_k}, \qquad i = 0, \dots, d_k/2 - 1
$$

Como os ponteiros de um relógio: pares rápidos distinguem posições próximas, pares lentos
distinguem posições distantes. O RoPE entra só em $Q$ e $K$, dentro de toda camada.

## 4. Multi-head e GQA

Cada linha do softmax produz uma única média ponderada. Para o token prestar atenção em várias
relações ao mesmo tempo, dividimos $d$ em $H$ cabeças:

$$
\text{head}_h = \text{Attention}(X W_Q^h, X W_K^h, X W_V^h), \qquad
\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)\, W_O
$$

O custo é o mesmo de uma cabeça grande: $H \cdot (d \cdot d/H) = d^2$. Concatenar e multiplicar
por $W_O$ é igual a $\sum_h \text{head}_h W_O^h$: cada cabeça escreve a sua parte.

**GQA:** $Q$ tem $H$ cabeças, $K$ e $V$ têm só $H_{kv}$, e grupos de $H/H_{kv}$ cabeças de $Q$
dividem o mesmo $K$ e $V$. O KV cache encolhe pelo fator $H/H_{kv}$.

## 5. Conexão residual

O bloco calcula $x + F(x)$. A derivada é

$$
\frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F}{\partial x_l}
$$

e o produto de $L$ termos $(I + J_l)$ sempre contém a identidade: existe um caminho direto da
perda até a primeira camada, e o gradiente não some. Cada camada lê do "fluxo residual" e soma
o seu resultado de volta.

## 6. Normalização

Somando saídas no fluxo residual, a escala dos vetores cresce e varia.

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta
\qquad\qquad
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}}
$$

A RMSNorm tira a centralização e é mais barata, com qualidade parecida. A propriedade que
importa: $\text{RMSNorm}(c\,x) = \text{RMSNorm}(x)$ para $c > 0$.

**Pre-norm.** O original fazia $\text{Norm}(x + F(x))$, o que põe a norma no caminho direto da
seção 5. O pre-norm deixa o caminho limpo:

$$
H = X + \text{MHA}(\text{Norm}(X)), \qquad X' = H + \text{FFN}(\text{Norm}(H))
$$

## 7. Feed-forward (FFN)

A atenção mistura informação entre tokens. O FFN processa cada token sozinho, com não
linearidade:

$$
\text{FFN}(x) = W_2\, \varphi(W_1 x), \qquad W_1: d \to 4d
$$

Leitura útil: cada linha de $W_1$ detecta um padrão, e cada coluna de $W_2$ é o que se escreve
quando ele aparece (uma memória de chave e valor).

**SwiGLU:**

$$
\text{FFN}(x) = W_2\big(\text{SiLU}(W_1 x) \odot W_3 x\big), \qquad \text{SiLU}(z) = z\,\sigma(z)
$$

Um caminho decide quanto passa, o outro leva o que passa. Com três matrizes, para manter
$8d^2$ parâmetros: $3 \cdot d \cdot h = 8d^2 \Rightarrow h = 8d/3$.

## 8. Saída e temperatura

$$
z = W_{\text{out}}\, h, \qquad P(\text{próximo} = i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

Com pesos amarrados, $W_{\text{out}} = E$. A temperatura age na razão entre probabilidades:

$$
\frac{p_i}{p_j} = e^{(z_i - z_j)/T}
$$

$T \to 0$ vira argmax; $T \to \infty$ vira uniforme.

## 9. Perda e o gradiente $p - y$

Maximizar $\prod_t P(x_{t+1} \mid x_{\le t})$ é o mesmo que minimizar

$$
\mathcal{L} = -\frac{1}{n} \sum_t \log P(x_{t+1} \mid x_1, \dots, x_t)
$$

que é a entropia cruzada entre o one-hot $y$ e a previsão $p$. Com $c$ sendo o token certo:

$$
\mathcal{L} = -z_c + \log \sum_j e^{z_j}
\qquad\Rightarrow\qquad
\frac{\partial \mathcal{L}}{\partial z_i} = -\delta_{ic} + \frac{e^{z_i}}{\sum_j e^{z_j}} = p_i - y_i
$$

O logit certo sobe com força $1 - p_c$; cada errado desce com força $p_i$.
Teste de sanidade: um modelo recém iniciado chuta uniforme, então a perda inicial é cerca de
$\ln V$.

## 10. Contas que caem em entrevista

- **Parâmetros por bloco:** $4d^2$ (atenção) $+ 8d^2$ (FFN) $= 12d^2$. Modelo:
  $\approx 12 L d^2 + V d$.
- **Custo da atenção:** $O(n^2 d)$, quadrático no contexto.
- **FLOPs por token:** $\approx 2P$ na inferência e $\approx 6P$ no treino ($P$ = parâmetros).
- **KV cache:** $2 \cdot L \cdot n \cdot H_{kv} \cdot d_k \cdot \text{bytes}$. No Llama 2 7B
  ($L = 32$, $H_{kv} = 32$, $d_k = 128$, fp16) são 0,5 MB por token, ou 2 GB para 4096 tokens.
