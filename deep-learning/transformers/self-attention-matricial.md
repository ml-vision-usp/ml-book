# Noção matricial de Self-Attention

Primeiramente, vamos considerar o caso sem Multi-headed attention. Assim, queremos computar os vetores contextualizados $$y_i, i = 1, \ldots , n$$ onde $$n$$ é o tamanho da sequencia (formada por tokens - ou palavras). Seja $$X_{n \times d}$$ a matriz de input. Essa matriz tem dimensão $$n \times d$$ onde $n$ é a quantidade de palavras na seuquência e $d$ é o tamanho do embedding. Podemos pensar que cada linha dessa matriz é o embedding de uma palavra.

Assim, como comentado anteriormente, usamos as palavras presentes na sequência de três maneiras distintas, associadas à três matrizes de pesos. Portanto, usaremos a seguinte notação:

- $Q_{(n \times d)} = X^Q_{(n \times d)} W^Q_{(d \times d)}$
- $K_{(n \times d)} = X^K_{(n \times d)} W^K_{(d \times d)}$
- $V_{(n \times d)} = X^V_{(n \times d)} W^V_{(d \times d)}$

onde $W^Q, W^K, W^V$ são matrizes de pesos treináveis. E $X^Q, X^K, X^V$ não passam de uma notação para diferenciar os três usos da matrix de entrada $X$.

Assim, primeiro relacionamos as palavras via dot product para obter os scores. Então, podemos representar esses scores como:
$$\text{Scores} = QK^T$$

Para entender melhor essa operação, usaremos a notação de colunas e linhas do Python, usando slicing. Então, se temos uma matriz $A$, sua i-ésima linha é $A[i, :]$ e sua j-ésima coluna é $A[:, j]$

Então, voltando à matriz $\text{Scores}$, a matriz $Q$ é usada no sentido de ser as palavras sobre as quais queremos computar self-attention (Query) e a matriz $K$ é usada no sentido de possuir as palavras com as quais, para cada palavra em $Q$, iremos computar a relação por meio do dot product. 

Então, se $Q[i, :]_{(1 \times d)}$ é uma palavra (vetor), $\text{Scores}[i, :]_{(1 \times n)}$ é importância contextual dessa palavra para todas as palavras da sequência.

Desse modo, perceba que $$\text{Scores}[i, :] = \begin{Bmatrix}
Q[i, :] \cdot K^T[:, 1] & Q[i, :] \cdot K^T[:, 2] & \cdots & Q[i, :] \cdot K^T[:, n]  \\
\end{Bmatrix}$$

Onde 
$$
A[i, :] \cdot K^T[:, j] = v_i \cdot v_j = s_{ij}
$$

No paper Attention is All you Need, há um fator de escala (que é um escalar) para essa multiplicação de matrizes. Esse fator não é tão importante para o entendimento conceitual desse mecanismo. Então, vamos ignorá-lo. 

Depois disso, devemos aplicar a função softmax para normalizar esses scores. Então, temos:

1. $\text{Scores} = QK^T$
2. $\text{softmax}\left( \text{Scores}\right)$ Os elementos dessa matriz são $w_{i,j}, i = 1, \ldots, n ; j = 1, \ldots, n$

Agora que já temos os scores normalizados que indicam a importância contextual (normalizada) de cada palavra relacionada entre si, entramos em um próximo estágio do processo:

O vetor contextualizado (${\bf y}_i$) de uma palavra ${\bf v}_i$ é a soma do produto de todas as palavras e os seus coeficientes de attention em relação à essa palavra ${\bf v}_i$.

Então para obter essa soma, usamos a matriz de palavras $V$. Essa matriz é denominada value, pois estamos usando o valor de cada palavra (após já termos os coeficientes de attention) para obter os vetores contextualizados.

Seja o vetor contextualizado ${\bf y}_i = \begin{Bmatrix} y_{i,1} & y_{i,2} & \cdots & y_{i,d}
\end{Bmatrix}$

onde

$$
y_{i,j} = \sum^n_{k = 1} w_{i,k} v_{k, j}
$$

Perceba que isso é equivalente à multiplicar a linha $i$ da matriz $\text{Scores}$ pela coluna $j$ da matriz $V$. Disso, temos:

$$
\text{Scores}_{(n \times n)}V_{(n \times d)} = C_{(n \times d)}
$$

Onde o $C[i,:] = {\bf y}_i$

Portanto, temos que:

$$
\text{Attention}\left(Q, K , V\right) = \text{softmax}\left(QK^T\right)V = C_{(n \times d)}
$$

## Multihead Attention
Como já comentado, multihead attention é usado ao fazer a entrada (no caso a matriz $X$) passar em paralelo por diversos mecanismos de self-attention. A ideia é que, dessa forma, consigamos capturar várias nuances de relação semântica entre as palavras (ou tokens). Desse modo, temos:

$$
\text{MultiHead}(X^Q, X^K, X^V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)W^O
$$

Onde:
$$\text{head}_i = \text{Attention}(X^QW^Q_i, X^KW^K_i, X^VW^V_i)$$

Observe a concatenação dos diversos resultados de self-attention e a passagem desse vetor concatenado por uma matriz de pesos:

$$
\text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)_{(n \times hd)}W^O_{(hd \times d)} = M_{(n \times d)}
$$

## Fator de escala de $QK^T$

No artigo Attention is All you Need, o mecanismo de Attention é definido como se segue:
$$
\text{Attention}\left(Q, K , V\right) = \text{softmax}\left(\dfrac{QK^T}{\sqrt{d_k}}\right)V
$$

O fator de escala 
$$
\dfrac{1}{\sqrt{d_k}}
$$

Onde $$d_k$$ é $$d$$, a dimensão do embedding dos tokens. 

A justificativa para a presença desse fator de escala na operação é para lidar com o problema de desaparecimento de gradiente relacionado ao dot product utilizado para computar attention. Perceba que essa escala é uma operação element wise na matriz $$\text{Scores}$$. Então, para ilustrar como os valores oriundos do dot product podem ficar grandes, o exemplo utilizado é o seguinte:

Sabendo que $$\text{Scores} = QK^T$$ é dado por:

$$\text{Scores}[i, :] = \begin{Bmatrix}
Q[i, :] \cdot K^T[:, 1] & Q[i, :] \cdot K^T[:, 2] & \cdots & Q[i, :] \cdot K^T[:, n]  \\
\end{Bmatrix}$$

Onde 

$$
Q[i, :] \cdot K^T[:, j] = q^{(i)} \cdot k^{(j)}
$$

Em que $$q^{(i)}$$ e $$k^{(j)}$$ são tokens cujo input tem dimensão $$d$$, esse dot product é dado por:

$$
q^{(i)} \cdot k^{(j)} = \sum_{p=1}^{d_k} q^{(i)}_p k^{(j)}_p
$$

Então, assumindo que os componentes de $$q^{(i)}$$ e $$k^{(j)}$$ sejam independentes entre si, tenham média 0 e variância 1, temos:

$$
\text{E} \left(q^{(i)} \cdot k^{(j)}\right) = \text{E} \left( \sum_{p=1}^{d_k} q^{(i)}_p k^{(j)}_p \right) = \sum_{p=1}^{d_k} \text{E} \left( q^{(i)}_p k^{(j)}_p \right) = \sum_{p=1}^{d_k} \text{E} \left( q^{(i)}_p \right) \text{E} \left( k^{(j)}_p \right) = 0                                                   
$$
e

$$
\text{Var} \left(q^{(i)} \cdot k^{(j)}\right) = \text{E} \left(\left(q^{(i)} \cdot k^{(j)} \right)^2\right) - \left(\text{E} \left(q^{(i)} \cdot k^{(j)}\right)\right)^2 = \text{E} \left(\left(q^{(i)} \cdot k^{(j)} \right)^2\right) 
$$

Disso,

$$
\text{E} \left(\left(q^{(i)} \cdot k^{(j)} \right)^2\right) = \text{E} \left( \sum_{p=1}^{d_k} q^{(i)}_p k^{(j)}_p  \sum_{r=1}^{d_k} q^{(i)}_r k^{(j)}_r \right) = \text{E} \left( \sum_{\substack{p=1 \\ r=1}}^{d_k} q^{(i)}_p k^{(j)}_p q^{(i)}_r k^{(j)}_r \right) =  \sum_{\substack{p=1 \\ r=1}}^{d_k} \text{E} \left( q^{(i)}_p k^{(j)}_p q^{(i)}_r k^{(j)}_r \right)
$$

Quando $$p \neq r$$:

$$
\text{E} \left( q^{(i)}_p k^{(j)}_p q^{(i)}_r k^{(j)}_r \right) = 0
$$
Pois os elementos de $$q$$ e $$k$$ têm média 0.
Agora, quando $$p = r$$:
$$
\text{E} \left( q^{(i)^2}_p k^{(j)^2}_p\right) = \text{E} \left( q^{(i)^2}_p \right) \text{E} \left(k^{(j)^2}_p\right) = \left( 0 - \text{E} \left( q^{(i)^2}_p \right) \right) \left(0 - \text{E} \left(k^{(j)^2}_p\right) \right)
$$

Então:
$$
\text{E} \left( q^{(i)^2}_p k^{(j)^2}_p\right) = \left( \text{E} \left( q^{(i)}_p \right) - \text{E} \left( q^{(i)^2}_p \right) \right) \left( \text{E} \left(k^{(j)}_p\right) - \text{E} \left(k^{(j)^2}_p\right) \right) = \text{Var} \left( q^{(i)}_p \right) \text{Var} \left( k^{(j)}_p \right) = 1 \cdot 1 = 1
$$

Assim, 
$$
\text{Var} \left(q^{(i)} \cdot k^{(j)}\right) = \sum_{\substack{p=1 \\ r=1}}^{d_k} \text{E} \left( q^{(i)}_p k^{(j)}_p q^{(i)}_r k^{(j)}_r \right) = \sum_{p=1}^{d_k} \text{E} \left( q^{(i)^2}_p k^{(j)^2}_p \right) = \sum_{p=1}^{d_k} 1 = d_k
$$

Com esse valor tão alto de variância a função Softmax pode nos devolver valores baixos demais, o que causa desaparecimento de gradiente. Ao usarmos o fator de escala, recuperamos uma variância de 1, o que melhora essa questão do desaparecimento do gradiente.

## Adicionando Batches à jogada

Quando falamos de deep learning, o método de otimização mais utilizado é Mini-Batch Gradient Descent. Então, como incorporar batches à esse processo de treinamento?

A resposta para essa pergunta não é tão complicada. Na prática, basta incorporar mais uma dimensão à matriz de entrada $X$. Então, para cada elemento nessa dimensão (que é uma sequência distinta de input), efetuamos as operações comentadas até aqui.
Então, a matriz de entrada é da forma:
$$
X_{(b \times n \times d)}
$$

## Reflexão sobre tamanho máximo de sequência
#### Co-escrito por João Gabriel
É interessante comentar sobre batches pois essa técnica, juntamente com Positional encoding (que será abordado mais adiante) implica em uma conhecida limitação da arquitetura Transformer. Perceba que, em um batch de 32 sequências, a matriz de entrada é tal que temos um valor constante (para cada sequência) de $n$ (tamanho da sequência) e de $d$ (tamanho do embedding dos tokens). 

Para $d$ isso é simples. Cada token deve ter o mesmo comprimento, isto é, devem ser vetores com a mesma quantidade de elementos.

Agora, para $n$, isso não é tão simples assim. Na prática, as sequências nesse batch não precisam ter o mesmo comprimento. O valor $n$  deve corrsponder ao tamanho da maior sequência no batch. Então, o restante das sequências são completadas com tokens especiais de padding, para chegar à esse tamanho.

Desse modo, introduzimos à noção de um tamanho máximo de sequência na arquitetura Transformer, mas qual é o maior tamanho de sequência que essa arquitetura é capaz de lidar. 

Se observarmos as matrizes de pesos treináveis $W^Q_{(d \times d)}, W^K_{(d \times d)}, W^V_{(d \times d)}$ é notável que elas não possuem influência de $n$. Então, no moecanismo de self-attention, não há uma limitação teórica para o tamanho da sequência. Obviamente teremos operações mais custosas ao computar

$$
\text{Attention}\left(Q, K , V\right) = \text{softmax}\left(QK^T\right)V
$$

Contudo, isso não é nenhum impeditivo. Porém, além desse custo elevado de processamento, um problema a se considerar ao lidar com sequências longas é o da memória. Perceba que no caso de uma sequência (de comprimento $n_0$) no batch ser consideravelmente mais longa que as demais, precisaremos ter uma matriz de entrada:

$$
X_{(b \times n_0 \times d)}
$$

Desse modo, temos complexidade $O(n_0)^2$ em memória sendo que como as demais sequências são bem menores, a maior parte dessa memória está sendo desperdiçada com tokens de padding.

Além dessas questões, sequências muito longas podem fazer com que o Positional-Encoding empregado no modelo não funcione bem. Já que geralmente esse encoding é composto de uma combinação de ondas senoidais, é possível que para sequências muito longas, o período dessas ondas não seja longo o suficiente, o que pode fazer com que uma palavra no final da sequência seja interpretada como uma palavra no início da sequência.