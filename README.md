<p align="justify"><h1>Mini Transformer em PyTorch</h1></p>

<p align="justify">
Este projeto implementa um <b>Mini Transformer autoregressivo</b> utilizando <b>PyTorch</b>, reproduzindo os principais componentes presentes em arquiteturas modernas de Large Language Models (LLMs), como embeddings, positional encoding senoidal, multi-head self-attention, feedforward networks e geração autoregressiva de texto.
</p>

<p align="justify">
O objetivo principal foi compreender profundamente o funcionamento interno de transformers modernos através da implementação prática de uma versão reduzida, porém funcional, capaz de aprender padrões linguísticos e gerar texto caractere por caractere após treinamento supervisionado.
</p>

---

<p align="justify"><h2>Objetivos do Projeto</h2></p>

<p align="justify">
Este projeto foi desenvolvido com os seguintes objetivos:
</p>

<ul>
<li>Compreender a arquitetura Transformer;</li>
<li>Implementar positional encoding senoidal;</li>
<li>Aplicar causal masking em geração autoregressiva;</li>
<li>Treinar um modelo generativo em nível de caracteres;</li>
<li>Entender o papel da self-attention;</li>
<li>Explorar geração sequencial token a token;</li>
<li>Criar base para futuros transformers com memória persistente.</li>
</ul>

---

<p align="justify"><h2>Importação das Bibliotecas</h2></p>

<p align="justify">
O projeto utiliza principalmente o framework <b>PyTorch</b>, responsável pela criação da arquitetura neural, gerenciamento automático de gradientes e treinamento em GPU utilizando CUDA.
</p>

```python
import torch
import torch.nn as nn
import math
```

<p align="justify">
As bibliotecas possuem as seguintes funções:
</p>

| Biblioteca | Função |
|---|---|
| torch | Operações tensoriais |
| torch.nn | Camadas neurais |
| math | Operações matemáticas |

---

<p align="justify"><h2>Configuração do Device</h2></p>

<p align="justify">
O código detecta automaticamente se existe uma GPU CUDA disponível. Caso exista, o treinamento é executado diretamente na GPU, acelerando significativamente o processamento das operações matriciais do transformer.
</p>

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

<p align="justify">
Isso permite que:
</p>

<ul>
<li>Tensores sejam processados em paralelo;</li>
<li>Attention seja computada mais rapidamente;</li>
<li>Treinamentos longos sejam viáveis;</li>
<li>Grandes multiplicações matriciais sejam aceleradas.</li>
</ul>

---

<p align="justify"><h2>Classe PositionalEncoding</h2></p>

<p align="justify">
A classe <b>PositionalEncoding</b> é responsável por adicionar informação posicional aos embeddings.
</p>

<p align="justify">
Transformers processam todos os tokens simultaneamente e, naturalmente, não possuem noção de ordem sequencial.
</p>

<p align="justify">
Sem positional encoding, as frases:
</p>

```text
gato bebe leite
```

<p align="justify">
e:
</p>

```text
leite gato bebe
```

<p align="justify">
poderiam parecer semanticamente equivalentes para o modelo.
</p>

<p align="justify">
Por isso, é necessário adicionar explicitamente informações sobre posição.
</p>

```python
class PositionalEncoding(nn.Module):
```

---

<p align="justify"><h2>Método __init__ da Classe PositionalEncoding</h2></p>

```python
def __init__(self, d_model, dropout=0.1, max_len=5000):
```

<p align="justify">
Esse método constrói toda a estrutura do positional encoding.
</p>

| Parâmetro | Função |
|---|---|
| d_model | Dimensão dos embeddings |
| dropout | Regularização |
| max_len | Tamanho máximo da sequência |

---

<p align="justify"><h2>Construção das Posições</h2></p>

```python
position = torch.arange(max_len).unsqueeze(1)
```

<p align="justify">
Essa linha cria:
</p>

```text
0
1
2
3
...
```

<p align="justify">
representando as posições dos tokens.
</p>

<p align="justify">
O método:
</p>

```python
unsqueeze(1)
```

<p align="justify">
adiciona uma dimensão extra necessária para broadcasting matricial.
</p>

---

<p align="justify"><h2>Frequências do Positional Encoding</h2></p>

```python
div_term = torch.exp(
    torch.arange(0, d_model, 2)
    * (-math.log(10000.0) / d_model)
)
```

<p align="justify">
Cada dimensão do embedding recebe uma frequência diferente.
</p>

<p align="justify">
Isso gera assinaturas posicionais únicas para cada token.
</p>

---

<p align="justify"><h2>Fórmulas do Positional Encoding</h2></p>

<p align="center">

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

</p>

<p align="center">

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

</p>

<p align="justify">
As dimensões pares utilizam seno e as ímpares utilizam cosseno.
</p>

<p align="justify">
Essas funções periódicas permitem que o transformer aprenda relações relativas entre posições.
</p>

---

<p align="justify"><h2>Registro do Buffer Posicional</h2></p>

```python
self.register_buffer('pe', pe)
```

<p align="justify">
Essa linha registra a matriz posicional como um buffer do modelo.
</p>

<p align="justify">
Buffers:
</p>

<ul>
<li>Não são parâmetros treináveis;</li>
<li>Mas acompanham GPU/CPU automaticamente;</li>
<li>São salvos junto ao modelo.</li>
</ul>

---

<p align="justify"><h2>Método forward da Classe PositionalEncoding</h2></p>

```python
def forward(self, x):
```

<p align="justify">
O método <b>forward</b> define o fluxo computacional da classe.
</p>

```python
x = x + self.pe[:, :x.size(1)]
```

<p align="justify">
Aqui ocorre a soma:
</p>

<ul>
<li>Embedding semântico;</li>
<li>Informação posicional.</li>
</ul>

<p align="justify">
O resultado é um embedding contextualizado espacialmente.
</p>

---

<p align="justify"><h2>Classe MiniTransformer</h2></p>

<p align="justify">
A classe <b>MiniTransformer</b> representa a arquitetura principal do modelo.
</p>

```python
class MiniTransformer(nn.Module):
```

<p align="justify">
Ela contém:
</p>

<ul>
<li>Embedding Layer;</li>
<li>Positional Encoding;</li>
<li>Transformer Encoder;</li>
<li>Camada Linear final.</li>
</ul>

---

<p align="justify"><h2>Método __init__ da Classe MiniTransformer</h2></p>

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

<p align="justify"><h2>Embedding Layer</h2></p>

```python
self.embedding = nn.Embedding(vocab_size, d_model)
```

<p align="justify">
Essa camada converte IDs inteiros em vetores densos.
</p>

<p align="justify">
Exemplo:
</p>

```text
15
↓
[0.12, -0.77, 0.91, ...]
```

<p align="justify">
Os embeddings representam semanticamente os tokens em espaço vetorial contínuo.
</p>

---

<p align="justify"><h2>Positional Encoder</h2></p>

```python
self.pos_encoder = PositionalEncoding(d_model, dropout)
```

<p align="justify">
Instancia o positional encoding anteriormente definido.
</p>

---

<p align="justify"><h2>TransformerEncoderLayer</h2></p>

```python
encoder_layers = nn.TransformerEncoderLayer(
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    batch_first=True
)
```

<p align="justify">
Esse bloco implementa:
</p>

<ul>
<li>Multi-head self-attention;</li>
<li>Feedforward network;</li>
<li>Residual connections;</li>
<li>Layer normalization.</li>
</ul>

<p align="justify">
O parâmetro:
</p>

```python
batch_first=True
```

<p align="justify">
define o formato:
</p>

```text
(batch, seq_len, features)
```

---

<p align="justify"><h2>Empilhamento dos Encoders</h2></p>

```python
self.transformer_encoder = nn.TransformerEncoder(
    encoder_layers,
    num_layers
)
```

<p align="justify">
Essa linha empilha múltiplos blocos transformer.
</p>

<p align="justify">
Exemplo:
</p>

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

<p align="justify"><h2>Camada Linear Final</h2></p>

```python
self.fc_out = nn.Linear(d_model, vocab_size)
```

<p align="justify">
Essa camada transforma embeddings contextuais em logits para cada token do vocabulário.
</p>

---

<p align="justify"><h2>Máscara Causal</h2></p>

```python
self.register_buffer('src_mask', None)
```

<p align="justify">
A máscara causal impede que o modelo enxergue tokens futuros durante treinamento autoregressivo.
</p>

---

<p align="justify"><h2>Método _generate_square_subsequent_mask</h2></p>

```python
def _generate_square_subsequent_mask(self, sz):
```

<p align="justify">
Esse método cria a máscara triangular superior utilizada no causal masking.
</p>

<p align="justify">
Matematicamente:
</p>

<p align="center">

$$
M_{ij} =
\begin{cases}
0 & j \le i \\
-\infty & j > i
\end{cases}
$$

</p>

<p align="justify">
O valor:
</p>

```text
-\infty
```

<p align="justify">
faz com que o softmax transforme essas posições em probabilidade zero.
</p>

---

<p align="justify"><h2>Self-Attention</h2></p>

<p align="justify">
A self-attention é o mecanismo central do transformer.
</p>

<p align="justify">
Ela permite que cada token observe todos os outros tokens da sequência.
</p>

<p align="center">

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

</p>

| Componente | Função |
|---|---|
| Q | Query |
| K | Key |
| V | Value |

<p align="justify">
O produto:
</p>

<p align="center">

$$
QK^T
$$

</p>

<p align="justify">
mede similaridade entre tokens.
</p>

---

<p align="justify"><h2>Método forward da Classe MiniTransformer</h2></p>

```python
def forward(self, src):
```

<p align="justify">
Esse método define todo o fluxo do transformer.
</p>

---

<p align="justify"><h2>Geração da Máscara</h2></p>

```python
if self.src_mask is None or self.src_mask.size(0) != src.size(1):
```

<p align="justify">
Verifica se a máscara causal precisa ser recriada.
</p>

---

<p align="justify"><h2>Embedding dos Tokens</h2></p>

```python
src = self.embedding(src) * math.sqrt(self.d_model)
```

<p align="justify">
Os tokens são convertidos em embeddings.
</p>

<p align="justify">
O fator:
</p>

```python
math.sqrt(self.d_model)
```

<p align="justify">
estabiliza variância e gradientes.
</p>

---

<p align="justify"><h2>Adição do Positional Encoding</h2></p>

```python
src = self.pos_encoder(src)
```

<p align="justify">
Combina:
</p>

<ul>
<li>conteúdo semântico;</li>
<li>posição dos tokens.</li>
</ul>

---

<p align="justify"><h2>Transformer Encoder</h2></p>

```python
output = self.transformer_encoder(
    src,
    mask=self.src_mask
)
```

<p align="justify">
Aqui ocorre:
</p>

<ul>
<li>self-attention;</li>
<li>troca contextual de informação;</li>
<li>processamento profundo da sequência.</li>
</ul>

<p align="justify">
Esse é o núcleo computacional do transformer.
</p>

---

<p align="justify"><h2>Camada de Saída</h2></p>

```python
output = self.fc_out(output)
```

<p align="justify">
Transforma embeddings contextuais em probabilidades para o próximo token.
</p>

<p align="justify">
O shape final é:
</p>

```text
(batch, seq_len, vocab_size)
```

---

<p align="justify"><h2>Dataset</h2></p>

<p align="justify">
Foi criado um pequeno dataset textual contendo frases relacionadas à inteligência artificial e transformers.
</p>

```python
with open("data.txt", "w", encoding="utf-8") as f:
    f.write(sample_text)
```

<p align="justify">
Mesmo pequeno, o dataset já permite aprendizado de:
</p>

<ul>
<li>estrutura linguística;</li>
<li>continuidade textual;</li>
<li>dependências locais;</li>
<li>padrões sintáticos.</li>
</ul>

---

<p align="justify"><h2>Tokenização Character-Level</h2></p>

<p align="justify">
O modelo opera em nível de caracteres.
</p>

```text
"IA"
↓
["I","A"]
↓
[12,5]
```

<p align="justify">
Isso simplifica:
</p>

<ul>
<li>vocabulário;</li>
<li>treinamento;</li>
<li>implementação do tokenizer.</li>
</ul>

---

<p align="justify"><h2>Mini-batches</h2></p>

```python
def get_batch():
```

<p align="justify">
Seleciona trechos aleatórios do texto.
</p>

| Input | Target |
|---|---|
| ABCDEF | BCDEFG |

<p align="justify">
O objetivo é prever o próximo caractere.
</p>

---

<p align="justify"><h2>Treinamento</h2></p>

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)
```

<p align="justify">
O AdamW é uma versão moderna do gradient descent utilizada em transformers.
</p>

```python
criterion = nn.CrossEntropyLoss()
```

<p align="justify">
A cross entropy mede distância entre:
</p>

<ul>
<li>distribuição prevista;</li>
<li>distribuição correta.</li>
</ul>

---

<p align="justify"><h2>Backpropagation</h2></p>

```python
loss.backward()
```

<p align="justify">
Calcula gradientes automaticamente.
</p>

```python
optimizer.step()
```

<p align="justify">
Atualiza os pesos da rede neural.
</p>

---

<p align="justify"><h2>Convergência da Loss</h2></p>

<p align="justify">
Durante o treinamento:
</p>

```text
4.12 → 0.09
```

<p align="justify">
A redução contínua da loss demonstra:
</p>

<ul>
<li>aprendizado estável;</li>
<li>convergência do modelo;</li>
<li>capacidade de aprender padrões linguísticos.</li>
</ul>

---

<p align="justify"><h2>Geração Autoregressiva</h2></p>

```python
logits = model(context)
```

<p align="justify">
O modelo gera logits para o próximo token.
</p>

```python
probs = torch.softmax(
    logits[:, -1, :],
    dim=-1
)
```

<p align="justify">
O softmax converte logits em probabilidades.
</p>

<p align="center">

$$
softmax(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

</p>

```python
next_token = torch.multinomial(
    probs,
    num_samples=1
)
```

<p align="justify">
O próximo token é amostrado probabilisticamente.
</p>

<p align="justify">
Isso torna a geração:
</p>

<ul>
<li>menos determinística;</li>
<li>mais criativa;</li>
<li>mais variada.</li>
</ul>

---

<p align="justify"><h2>Resultado Obtido</h2></p>

```text
A inteligência artificial está avançando rapidamente e é fascina
```

<p align="justify">
Mesmo utilizando um dataset pequeno, o modelo conseguiu aprender:
</p>

<ul>
<li>continuidade textual;</li>
<li>estrutura sintática;</li>
<li>dependências contextuais;</li>
<li>padrões linguísticos.</li>
</ul>

---

<p align="justify"><h2>Fluxo Completo da Arquitetura</h2></p>

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

<p align="justify"><h2>Conclusão</h2></p>

<p align="justify">
Este projeto demonstrou na prática os principais mecanismos internos de transformers modernos através da implementação de um Mini Transformer autoregressivo em PyTorch.
</p>

<p align="justify">
Mesmo sendo uma versão reduzida, a arquitetura implementada reproduz componentes fundamentais presentes em modelos de larga escala:
</p>

<ul>
<li>Embeddings;</li>
<li>Positional Encoding;</li>
<li>Self-Attention;</li>
<li>Causal Masking;</li>
<li>Feedforward Networks;</li>
<li>Treinamento autoregressivo.</li>
</ul>

<p align="justify">
O experimento demonstrou que transformers são essencialmente sistemas de otimização baseados em operações matriciais massivas guiadas por atenção contextual e aprendizado por gradiente.
</p>

<p align="justify">
Além disso, este projeto estabelece uma base sólida para futuras extensões envolvendo:
</p>

<ul>
<li>Memória persistente;</li>
<li>Retrieval-Augmented Generation (RAG);</li>
<li>KV Cache;</li>
<li>Transformers híbridos;</li>
<li>Arquiteturas cognitivas com memória temporal.</li>
</ul>
