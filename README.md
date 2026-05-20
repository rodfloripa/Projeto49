````md
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
````

<p align="justify">
As bibliotecas possuem as seguintes funções:
</p>

| Biblioteca | Função                |
| ---------- | --------------------- |
| torch      | Operações tensoriais  |
| torch.nn   | Camadas neurais       |
| math       | Operações matemáticas |

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

<p align="justify"><h2>Positional Encoding</h2></p>

<p align="justify">
Transformers não possuem noção natural de ordem sequencial. Diferentemente de RNNs e LSTMs, a arquitetura Transformer processa todos os tokens simultaneamente.
</p>

<p align="justify">
Por isso, é necessário injetar explicitamente informações posicionais nos embeddings.
</p>

<p align="justify">
O projeto utiliza o positional encoding senoidal introduzido no paper clássico:
</p>

<p align="center">

[
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
]

</p>

<p align="center">

[
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
]

</p>

<p align="justify">
A principal vantagem desse método é permitir que o modelo aprenda relações relativas entre posições utilizando funções periódicas suaves.
</p>

```python
class PositionalEncoding(nn.Module):
```

<p align="justify">
A classe constrói uma matriz contendo:
</p>

<ul>
<li>Seno nas posições pares;</li>
<li>Cosseno nas posições ímpares;</li>
<li>Frequências diferentes para cada dimensão.</li>
</ul>

---

<p align="justify"><h2>Construção da Matriz Posicional</h2></p>

```python
position = torch.arange(max_len).unsqueeze(1)
```

<p align="justify">
Essa linha cria os índices de posição:
</p>

```text
0
1
2
3
...
```

<p align="justify">
Posteriormente:
</p>

```python
div_term = torch.exp(
    torch.arange(0, d_model, 2)
    * (-math.log(10000.0) / d_model)
)
```

<p align="justify">
são geradas as frequências utilizadas nas funções seno e cosseno.
</p>

---

<p align="justify"><h2>Aplicação do Seno e Cosseno</h2></p>

```python
pe[0, :, 0::2] = torch.sin(position * div_term)
pe[0, :, 1::2] = torch.cos(position * div_term)
```

<p align="justify">
As dimensões pares recebem seno e as ímpares recebem cosseno.
</p>

<p align="justify">
Isso gera padrões periódicos únicos para cada posição da sequência.
</p>

---

<p align="justify"><h2>Embeddings</h2></p>

<p align="justify">
Transformers não trabalham diretamente com texto bruto.
</p>

<p align="justify">
Cada caractere é convertido em um índice inteiro:
</p>

```python
stoi = {ch:i for i, ch in enumerate(chars)}
```

<p align="justify">
Posteriormente, esses índices são transformados em vetores densos:
</p>

```python
self.embedding = nn.Embedding(vocab_size, d_model)
```

<p align="justify">
Esses embeddings representam semanticamente os caracteres em um espaço vetorial contínuo.
</p>

---

<p align="justify"><h2>Transformer Encoder</h2></p>

<p align="justify">
O núcleo do modelo é construído utilizando:
</p>

```python
nn.TransformerEncoderLayer
```

<p align="justify">
Cada bloco encoder contém:
</p>

<ul>
<li>Multi-head self-attention;</li>
<li>Residual connections;</li>
<li>Layer normalization;</li>
<li>Feedforward network.</li>
</ul>

---

<p align="justify"><h2>Self-Attention</h2></p>

<p align="justify">
A self-attention é o mecanismo responsável por permitir que cada token observe todos os outros tokens da sequência.
</p>

<p align="justify">
A fórmula principal da attention é:
</p>

<p align="center">

[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
]

</p>

<p align="justify">
Onde:
</p>

| Componente | Função |
| ---------- | ------ |
| Q          | Query  |
| K          | Key    |
| V          | Value  |

<p align="justify">
O produto:
</p>

<p align="center">

[
QK^T
]

</p>

<p align="justify">
mede similaridade entre tokens.
</p>

---

<p align="justify"><h2>Causal Masking</h2></p>

<p align="justify">
Como o modelo é autoregressivo, ele não pode enxergar tokens futuros durante o treinamento.
</p>

<p align="justify">
Por isso, foi implementada uma máscara causal:
</p>

```python
_generate_square_subsequent_mask
```

<p align="justify">
Essa máscara impede vazamento de informação futura.
</p>

<p align="justify">
Matematicamente:
</p>

<p align="center">

[
M_{ij} =
\begin{cases}
0 & j \le i \
-\infty & j > i
\end{cases}
]

</p>

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
Embora pequeno, o dataset já permite que o modelo aprenda:
</p>

<ul>
<li>Estruturas linguísticas;</li>
<li>Padrões sintáticos;</li>
<li>Dependências locais;</li>
<li>Continuidade textual.</li>
</ul>

---

<p align="justify"><h2>Tokenização Character-Level</h2></p>

<p align="justify">
O modelo opera em nível de caracteres.
</p>

<p align="justify">
Exemplo:
</p>

```text
"IA"
↓
["I", "A"]
↓
[12, 5]
```

<p align="justify">
Isso simplifica:
</p>

<ul>
<li>Vocabulário;</li>
<li>Treinamento;</li>
<li>Estrutura do tokenizer.</li>
</ul>

---

<p align="justify"><h2>Mini-batches</h2></p>

```python
def get_batch():
```

<p align="justify">
Essa função seleciona trechos aleatórios do texto para treinamento.
</p>

<p align="justify">
A entrada:
</p>

```text
ABCDEFG
```

<p align="justify">
gera:
</p>

| Input  | Target |
| ------ | ------ |
| ABCDEF | BCDEFG |

<p align="justify">
O objetivo é prever o próximo caractere.
</p>

---

<p align="justify"><h2>Treinamento</h2></p>

<p align="justify">
O treinamento utiliza:
</p>

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)
```

<p align="justify">
O AdamW é uma variante moderna do gradient descent, amplamente utilizada em transformers.
</p>

<p align="justify">
A loss utilizada foi:
</p>

```python
criterion = nn.CrossEntropyLoss()
```

<p align="justify">
A cross entropy mede quão distante a distribuição prevista está da distribuição correta.
</p>

---

<p align="justify"><h2>Backpropagation</h2></p>

<p align="justify">
O aprendizado ocorre através de:
</p>

```python
loss.backward()
```

<p align="justify">
O PyTorch calcula automaticamente:
</p>

<ul>
<li>Gradientes;</li>
<li>Derivadas parciais;</li>
<li>Atualizações dos pesos.</li>
</ul>

<p align="justify">
Posteriormente:
</p>

```python
optimizer.step()
```

<p align="justify">
atualiza os parâmetros da rede neural.
</p>

---

<p align="justify"><h2>Convergência da Loss</h2></p>

<p align="justify">
Durante o treinamento, a loss caiu de aproximadamente:
</p>

```text
4.12 → 0.09
```

<p align="justify">
Isso demonstra que o transformer conseguiu aprender padrões presentes no dataset.
</p>

<p align="justify">
A redução contínua da loss indica:
</p>

<ul>
<li>Aprendizado estável;</li>
<li>Convergência do modelo;</li>
<li>Capacidade de generalização local.</li>
</ul>

---

<p align="justify"><h2>Geração Autoregressiva</h2></p>

<p align="justify">
Após o treinamento, o modelo gera texto token por token.
</p>

```python
logits = model(context)
```

<p align="justify">
O transformer produz probabilidades para o próximo caractere.
</p>

<p align="justify">
Em seguida:
</p>

```python
probs = torch.softmax(
    logits[:, -1, :],
    dim=-1
)
```

<p align="justify">
transforma os logits em probabilidades.
</p>

<p align="justify">
O próximo token é amostrado utilizando:
</p>

```python
torch.multinomial(probs, num_samples=1)
```

<p align="justify">
Isso permite geração estocástica e evita respostas completamente determinísticas.
</p>

---

<p align="justify"><h2>Resultado Obtido</h2></p>

```text
A inteligência artificial está avançando rapidamente e é fascina
```

<p align="justify">
Mesmo utilizando um dataset extremamente pequeno, o modelo conseguiu aprender:
</p>

<ul>
<li>Estrutura sintática;</li>
<li>Continuidade textual;</li>
<li>Padrões linguísticos;</li>
<li>Dependências contextuais.</li>
</ul>

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
O experimento demonstrou que transformers são essencialmente sistemas de otimização sobre grandes operações matriciais guiadas por atenção contextual e aprendizado por gradiente.
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
```
