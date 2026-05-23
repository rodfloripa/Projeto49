
# 1. Mini Transformer em PyTorch

<div align="justify">

Este projeto implementa um <b>Mini Transformer autoregressivo</b> utilizando <b>PyTorch</b>, reproduzindo os principais componentes presentes em arquiteturas modernas de Large Language Models (LLMs), como embeddings, positional encoding senoidal, multi-head self-attention, feedforward networks e geração autoregressiva de texto.

O objetivo principal foi compreender profundamente o funcionamento interno de transformers modernos através da implementação prática de uma versão reduzida, porém funcional, capaz de aprender padrões linguísticos e gerar texto caractere por caractere após treinamento supervisionado.

</div>

---

# 2. Objetivos do Projeto

<div align="justify">

Este projeto foi desenvolvido com os seguintes objetivos:

<ul>
<li>Compreender a arquitetura Transformer</li>
<li>Implementar positional encoding senoidal</li>
<li>Aplicar causal masking em geração autoregressiva</li>
<li>Treinar um modelo generativo em nível de caracteres</li>
<li>Entender o papel da self-attention</li>
<li>Explorar geração sequencial token a token</li>
<li>Criar base para futuros transformers com memória persistente</li>
</ul>

</div>

---

# 3. Importação das Bibliotecas

<div align="justify">

O projeto utiliza principalmente o framework <b>PyTorch</b>, responsável pela criação da arquitetura neural, gerenciamento automático de gradientes e treinamento em GPU utilizando CUDA.

</div>

```python
import torch
import torch.nn as nn
import math
```

<div align="justify">

As bibliotecas possuem as seguintes funções:

</div>

| Biblioteca | Função |
|---|---|
| torch | Operações tensoriais |
| torch.nn | Camadas neurais |
| math | Operações matemáticas |

---

# 4. Configuração do Device

<div align="justify">

O código detecta automaticamente se existe uma GPU CUDA disponível. Caso exista, o treinamento é executado diretamente na GPU, acelerando significativamente o processamento das operações matriciais do transformer.

</div>

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

<div align="justify">

Isso permite que:

<ul>
<li>Tensores sejam processados em paralelo</li>
<li>Attention seja computada mais rapidamente</li>
<li>Treinamentos longos sejam viáveis</li>
<li>Grandes multiplicações matriciais sejam aceleradas</li>
</ul>

</div>

---

# 5. Classe PositionalEncoding

<div align="justify">

A classe <b>PositionalEncoding</b> é responsável por adicionar informação posicional aos embeddings.

Transformers processam todos os tokens simultaneamente e, naturalmente, não possuem noção de ordem sequencial.

Sem positional encoding, as frases:

</div>

```text
gato bebe leite
```

<div align="justify">

e:

</div>

```text
leite gato bebe
```

<div align="justify">

poderiam parecer semanticamente equivalentes para o modelo.

Por isso, é necessário adicionar explicitamente informações sobre posição.

</div>

```python
class PositionalEncoding(nn.Module):
```

---

# 6. Método __init__ da Classe PositionalEncoding

```python
def __init__(self, d_model, dropout=0.1, max_len=5000):
```

<div align="justify">

Esse método constrói toda a estrutura do positional encoding.

</div>

| Parâmetro | Função |
|---|---|
| d_model | Dimensão dos embeddings |
| dropout | Regularização |
| max_len | Tamanho máximo da sequência |

---

# 7. Construção das Posições

```python
position = torch.arange(max_len).unsqueeze(1)
```

<div align="justify">

Essa linha cria:

</div>

```text
0
1
2
3
...
```

<div align="justify">

representando as posições dos tokens.

O método:

</div>

```python
unsqueeze(1)
```

<div align="justify">

adiciona uma dimensão extra necessária para broadcasting matricial.

</div>

---

# 8. Frequências do Positional Encoding

```python
div_term = torch.exp(
    torch.arange(0, d_model, 2)
    * (-math.log(10000.0) / d_model)
)
```

<div align="justify">

Cada dimensão do embedding recebe uma frequência diferente.

Isso gera assinaturas posicionais únicas para cada token.

</div>

---

# 9. Fórmulas do Positional Encoding

```math
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
```

```math
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
```

<div align="justify">

As dimensões pares utilizam seno e as ímpares utilizam cosseno.

Essas funções periódicas permitem que o transformer aprenda relações relativas entre posições.

</div>

---

# 10. Registro do Buffer Posicional

```python
self.register_buffer('pe', pe)
```

<div align="justify">

Essa linha registra a matriz posicional como um buffer do modelo.

Buffers:

<ul>
<li>Não são parâmetros treináveis</li>
<li>Mas acompanham GPU/CPU automaticamente</li>
<li>São salvos junto ao modelo</li>
</ul>

</div>

---

# 11. Método forward da Classe PositionalEncoding

```python
def forward(self, x):
```

<div align="justify">

O método <b>forward</b> define o fluxo computacional da classe.

</div>

```python
x = x + self.pe[:, :x.size(1)]
```

<div align="justify">

Aqui ocorre a soma:

<ul>
<li>Embedding semântico</li>
<li>Informação posicional</li>
</ul>

O resultado é um embedding contextualizado espacialmente.

</div>

---

# 12. Classe MiniTransformer

<div align="justify">

A classe <b>MiniTransformer</b> representa a arquitetura principal do modelo.

</div>

```python
class MiniTransformer(nn.Module):
```

<div align="justify">

Ela contém:

<ul>
<li>Embedding Layer</li>
<li>Positional Encoding</li>
<li>Transformer Encoder</li>
<li>Camada Linear final</li>
</ul>

</div>

---

# 13. Método __init__ da Classe MiniTransformer

```python
def __init__(
    self,
    vocab_size,
    d_model,
    nhead,
    num_layers,
    dim_feedforward=2048,
    dropout=0.1
):
```

| Parâmetro | Função |
|---|---|
| vocab_size | Quantidade de tokens |
| d_model | Dimensão dos embeddings |
| nhead | Número de attention heads |
| num_layers | Número de blocos transformer |
| dim_feedforward | Tamanho da MLP interna |
| dropout | Regularização |

---

# 14. Embedding Layer

```python
self.embedding = nn.Embedding(vocab_size, d_model)
```

<div align="justify">

Essa camada converte IDs inteiros em vetores densos.

Exemplo:

</div>

```text
15
↓
[0.12, -0.77, 0.91, ...]
```

<div align="justify">

Os embeddings representam semanticamente os tokens em espaço vetorial contínuo.

</div>

---

# 15. Positional Encoder

```python
self.pos_encoder = PositionalEncoding(d_model, dropout)
```

<div align="justify">

Instancia o positional encoding anteriormente definido.

</div>

---

# 16. TransformerEncoderLayer

```python
encoder_layers = nn.TransformerEncoderLayer(
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    batch_first=True
)
```

<div align="justify">

Esse bloco implementa:

<ul>
<li>Multi-head self-attention</li>
<li>Feedforward network</li>
<li>Residual connections</li>
<li>Layer normalization</li>
</ul>

O parâmetro:

</div>

```python
batch_first=True
```

<div align="justify">

define o formato:

</div>

```text
(batch, seq_len, features)
```

---

# 17. Empilhamento dos Encoders

```python
self.transformer_encoder = nn.TransformerEncoder(
    encoder_layers,
    num_layers
)
```

<div align="justify">

Essa linha empilha múltiplos blocos transformer.

Exemplo:

</div>

```text
Transformer
↓
Transformer
↓
Transformer
↓
Transformer
```

---

# 18. Camada Linear Final

```python
self.fc_out = nn.Linear(d_model, vocab_size)
```

<div align="justify">

Essa camada transforma embeddings contextuais em logits para cada token do vocabulário.

</div>

---

# 19. Máscara Causal

```python
self.register_buffer('src_mask', None)
```

<div align="justify">

A máscara causal impede que o modelo enxergue tokens futuros durante treinamento autoregressivo.

</div>

---

# 20. Método _generate_square_subsequent_mask

```python
def _generate_square_subsequent_mask(self, sz):
```

<div align="justify">

Esse método cria a máscara triangular superior utilizada no causal masking.

Matematicamente:

</div>

```math
M_{ij} =
\begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
```

<div align="justify">

O valor:

</div>

```text
-\infty
```

<div align="justify">

faz com que o softmax transforme essas posições em probabilidade zero.

</div>

---

# 21. Self-Attention

<div align="justify">

A self-attention é o mecanismo central do transformer.

Ela permite que cada token observe todos os outros tokens da sequência.

</div>

```math
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

| Componente | Função |
|---|---|
| Q | Query |
| K | Key |
| V | Value |

<div align="justify">

O produto:

</div>

```math
QK^T
```

<div align="justify">

mede similaridade entre tokens.

</div>

---

# 22. Método forward da Classe MiniTransformer

```python
def forward(self, src):
```

<div align="justify">

Esse método define todo o fluxo do transformer.

</div>

---

# 23. Geração da Máscara

```python
if self.src_mask is None or self.src_mask.size(0) != src.size(1):
```

<div align="justify">

Verifica se a máscara causal precisa ser recriada.

</div>

---

# 24. Embedding dos Tokens

```python
src = self.embedding(src) * math.sqrt(self.d_model)
```

<div align="justify">

Os tokens são convertidos em embeddings.

O fator:

</div>

```python
math.sqrt(self.d_model)
```

<div align="justify">

estabiliza variância e gradientes.

</div>

---

# 25. Adição do Positional Encoding

```python
src = self.pos_encoder(src)
```

<div align="justify">

Combina:

<ul>
<li>conteúdo semântico</li>
<li>posição dos tokens</li>
</ul>

</div>

---

# 26. Transformer Encoder

```python
output = self.transformer_encoder(
    src,
    mask=self.src_mask
)
```

<div align="justify">

Aqui ocorre:

<ul>
<li>self-attention</li>
<li>troca contextual de informação</li>
<li>processamento profundo da sequência</li>
</ul>

Esse é o núcleo computacional do transformer.

</div>

---

# 27. Camada de Saída

```python
output = self.fc_out(output)
```

<div align="justify">

Transforma embeddings contextuais em probabilidades para o próximo token.

O shape final é:

</div>

```text
(batch, seq_len, vocab_size)
```

---

# 28. Dataset

<div align="justify">

Foi criado um pequeno dataset textual contendo frases relacionadas à inteligência artificial e transformers.

</div>

```python
with open("data.txt", "w", encoding="utf-8") as f:
    f.write(sample_text)
```

<div align="justify">

Mesmo pequeno, o dataset já permite aprendizado de:

<ul>
<li>estrutura linguística</li>
<li>continuidade textual</li>
<li>dependências locais</li>
<li>padrões sintáticos</li>
</ul>

</div>

---

# 29. Tokenização Character-Level

<div align="justify">

O modelo opera em nível de caracteres.

</div>

```text
"IA"
↓
["I","A"]
↓
[12,5]
```

<div align="justify">

Isso simplifica:

<ul>
<li>vocabulário</li>
<li>treinamento</li>
<li>implementação do tokenizer</li>
</ul>

</div>

---

# 30. Mini-batches

```python
def get_batch():
```

<div align="justify">

Seleciona trechos aleatórios do texto.

</div>

| Input | Target |
|---|---|
| ABCDEF | BCDEFG |

<div align="justify">

O objetivo é prever o próximo caractere.

</div>

---

# 31. Treinamento

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)
```

<div align="justify">

O AdamW é uma versão moderna do gradient descent utilizada em transformers.

</div>

```python
criterion = nn.CrossEntropyLoss()
```

<div align="justify">

A cross entropy mede distância entre:

<ul>
<li>distribuição prevista</li>
<li>distribuição correta</li>
</ul>

</div>

---

# 32. Backpropagation

```python
loss.backward()
```

<div align="justify">

Calcula gradientes automaticamente.

</div>

```python
optimizer.step()
```

<div align="justify">

Atualiza os pesos da rede neural.

</div>

---

# 33. Convergência da Loss

<div align="justify">

Durante o treinamento:

</div>

```text
4.12 → 0.09
```

<div align="justify">

A redução contínua da loss demonstra:

<ul>
<li>aprendizado estável</li>
<li>convergência do modelo</li>
<li>capacidade de aprender padrões linguísticos</li>
</ul>

</div>

---

# 34. Geração Autoregressiva

```python
logits = model(context)
```

<div align="justify">

O modelo gera logits para o próximo token.

</div>

```python
probs = torch.softmax(
    logits[:, -1, :],
    dim=-1
)
```

<div align="justify">

O softmax converte logits em probabilidades.

</div>

```math
softmax(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
```

```python
next_token = torch.multinomial(
    probs,
    num_samples=1
)
```

<div align="justify">

O próximo token é amostrado probabilisticamente.

Isso torna a geração:

<ul>
<li>menos determinística</li>
<li>mais criativa</li>
<li>mais variada</li>
</ul>

</div>

---

# 35. Resultado Obtido

```text
A inteligência artificial está avançando rapidamente e é fascina
```

<div align="justify">

Mesmo utilizando um dataset pequeno, o modelo conseguiu aprender:

<ul>
<li>continuidade textual</li>
<li>estrutura sintática</li>
<li>dependências contextuais</li>
<li>padrões linguísticos</li>
</ul>

</div>

---

# 36. Fluxo Completo da Arquitetura

```text
Tokens
↓
Embedding
↓
Positional Encoding
↓
Transformer Encoder
↓
Self-Attention
↓
FeedForward
↓
Camada Linear
↓
Probabilidades
↓
Próximo Token
```

---

# 37. Conclusão

<div align="justify">

Este projeto demonstrou na prática os principais mecanismos internos de transformers modernos através da implementação de um Mini Transformer autoregressivo em PyTorch.

Mesmo sendo uma versão reduzida, a arquitetura implementada reproduz componentes fundamentais presentes em modelos de larga escala:

<ul>
<li>Embeddings</li>
<li>Positional Encoding</li>
<li>Self-Attention</li>
<li>Causal Masking</li>
<li>Feedforward Networks</li>
<li>Treinamento autoregressivo</li>
</ul>

O experimento demonstrou que transformers são essencialmente sistemas de otimização baseados em operações matriciais massivas guiadas por atenção contextual e aprendizado por gradiente.

Além disso, este projeto estabelece uma base sólida para futuras extensões envolvendo:

<ul>
<li>Memória persistente</li>
<li>Retrieval-Augmented Generation (RAG)</li>
<li>KV Cache</li>
<li>Transformers híbridos</li>
<li>Arquiteturas cognitivas com memória temporal</li>
</ul>

</div>
````
