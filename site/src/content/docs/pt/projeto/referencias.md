---
title: Artigos e mapa de referências
description: As fontes primárias, suas relações com o código e as diferenças entre os artigos e este Transformer didático.
---

## Como usar esta bibliografia

Os artigos abaixo explicam conceitos presentes no motor. Cada registro traz link primário, autores completos, ano da primeira versão no arXiv, páginas relacionadas e uma citação BibTeX do preprint. A nota de publicação distingue eventos e periódicos quando essa informação aparece no registro primário. Por exemplo, Adam tem preprint de 2014 e publicação em ICLR 2015; usar o ano do preprint não altera essa data de publicação.

Use os links de páginas para partir de uma ideia e chegar à operação executável. A bibliografia identifica procedência conceitual; não significa que este projeto reproduz a arquitetura, o conjunto de dados ou os resultados experimentais de cada artigo.

## Mapa do artigo ao código

| Fonte | Conceito usado | Onde ler no código |
| --- | --- | --- |
| Attention Is All You Need | Atenção escalada e estrutura Transformer | `nn.Attention`, `nn.Block`, `model.GlassLM` |
| Neural Machine Translation of Rare Words with Subword Units | Fusões BPE para subpalavras | `tokenizer.Tokenizer.train`, `Tokenizer.encode` |
| SentencePiece | Contexto para tokenização e representação do espaço | `tokenizer.pre_tokenize`, `SPACE` |
| RoFormer | Rotações posicionais em consultas e chaves | `nn.rope_angles`, `autograd.rope` |
| GQA | Compartilhamento de cabeças K/V | `nn.Attention`, `autograd.repeat_kv` |
| Root Mean Square Layer Normalization | Normalização pela raiz da média quadrática | `nn.RMSNorm` |
| On Layer Normalization in the Transformer Architecture | Posição pre-norm | `nn.Block` |
| Deep Residual Learning for Image Recognition | Contexto das somas residuais | `nn.Block`, etapas `resid.attn` e `resid.ffn` |
| GLU Variants Improve Transformer | FFN com portão SwiGLU | `nn.SwiGLU`, `autograd.silu` |
| Using the Output Embedding to Improve Language Models | Pesos de embedding atrelados à saída | `GlassLM.E`, `GlassLM.forward` |
| Automatic differentiation in machine learning: a survey | Modo reverso e regra da cadeia | `autograd.Tensor.backward` |
| Adam: A Method for Stochastic Optimization | Momentos e atualização adaptativa | `optim.AdamW.step` |
| Decoupled Weight Decay Regularization | Decaimento de pesos desacoplado | `optim.AdamW.step` |
| SGDR: Stochastic Gradient Descent with Warm Restarts | Parte cosseno do schedule | `optim.lr_schedule` |

Os prefixos dos módulos nesta tabela são relativos a `glassllm`. A [referência pública](../../biblioteca/api-publica/) distingue exports da raiz e funções acessíveis por submódulo.

## A arquitetura é uma combinação própria

O Transformer original de [Vaswani et al.](https://arxiv.org/abs/1706.03762) tem encoder e decoder para transdução de sequências. O glassllm implementa um decoder causal pequeno para prever o próximo token; não inclui encoder nem atenção cruzada. Usa RoPE, RMSNorm, GQA e SwiGLU, escolhas introduzidas ou estudadas em trabalhos diferentes.

A implementação GQA compartilha duas cabeças K/V entre quatro cabeças Q. Não executa o procedimento de conversão e continuação de treino de checkpoints do [artigo GQA](https://arxiv.org/abs/2305.13245). O tokenizer é uma implementação própria de BPE sobre texto normalizado; `SentencePiece` é referência de contexto e não uma dependência nem garantia de compatibilidade de formato.

## Normalização e resíduos

O [artigo de pre-norm](https://arxiv.org/abs/2002.04745) analisa LayerNorm; o código usa RMSNorm antes das duas subcamadas e na saída final. Essa relação informa a localização da normalização, sem transferir automaticamente todas as conclusões teóricas entre operações distintas.

O artigo de [RMSNorm](https://arxiv.org/abs/1910.07467) também discute pRMSNorm; o projeto calcula a média quadrática de todas as dimensões. Com `ε > 0`, invariância de escala não é uma igualdade exata para todo vetor. O trabalho de [redes residuais](https://arxiv.org/abs/1512.03385) fornece contexto histórico de visão computacional, não uma prova de que este modelo mantém qualquer gradiente constante.

## Treino e otimização

AdamW usa momentos de gradientes e decaimento separado da atualização adaptativa. O schedule do projeto combina warmup com um decaimento cosseno até uma fração da taxa inicial. Ele **não implementa warm restarts** de SGDR, e o otimizador é AdamW em vez do SGD avaliado naquele artigo.

O autograd é próprio e opera sobre arrays NumPy. O [survey de diferenciação automática](https://arxiv.org/abs/1502.05767) fornece a terminologia e o panorama de métodos; os detalhes de acumulação, broadcasting e travessia pertencem à implementação descrita na [referência Tensor](../../biblioteca/tensor-autograd/).

## Citar e verificar

Abra o registro arXiv para verificar versões, PDF e detalhes de publicação. Os BibTeX desta página citam os preprints de forma consistente e incluem autores, título, ano, eprint, DOI arXiv e URL. O arquivo para download contém os 14 registros. Para um trabalho acadêmico que exija a versão publicada, use a referência dos anais ou periódico correspondente e declare qual versão foi consultada.

As equações dos fundamentos também explicitam formas e escolhas do código. Comece por [atenção causal](../../fundamentos/atencao-causal/) e [perda e gradientes](../../fundamentos/perda-gradientes/) para conectar um artigo à conta numérica executada.
