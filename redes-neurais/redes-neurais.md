# Redes Neurais

## Introdução

Redes Neurais são um modelo computacional composto por camadas de neurônios artificiais, também conhecidos como unidades de processamento, que são interconectados com o objetivo de aprender tarefas complexas no tocante a processamento de informação. 

Os **neurônios** são responsáveis por receberem entradas (inputs), realizarem operações matemáticas sobre elas, e então gerarem uma saída (output). Nesse sentido, cada neurônio está associado  a um certo número de **pesos** (weights) que o ligam à camada anterior, e também estão associados a um outro conjunto de pesos que o ligam à camada posterior, cada neurônio também passui um **viés** (bias) associado. Os pesos e viéses são o que aprendemos durante o processo de treinamento.

## Resumo

Antes de adentrar cada passo do processo de modo mais detalhado, primeiro será apresentado um resumo de todo o processo. Desse modo o leitor terá um panorama geral prévio antes de começar a ver cada tópico.

Para começar, temos que saber que cada neurônio recebe como entrada uma soma ponderada pelos pesos e viéses sobre as saídas da camada anterior. O resultado dessa soma é então aplicada em uma **função de ativação**, com a saída dessa função sendo também a saída do neurônio. As funções de ativação são utilizadas nos neurônios para introduzir não=linearidade nas saídas das redes neurais, algumas das funções comumente utilizadas são a sigmoide, a ReLU, e a tangente hiperbólica. A escolha da função de ativação pode afetar a performance do modelo.

Assim sendo, o **forward propagation** é o primeiro passo do processo de aprendizagem, nele os dados fluem da **camada de entrada** (input layer) para a última camada passando pelas **camadas escondidas** (hidden layers). Cada um dos neurônios da primeira camada escondida recebe como entrada uma soma ponderada pelos pesos sobre os dados de entrada e o seu viés, e então as saídas de cada um desses neurônios, o que chamamos de **sinal**, são passadas para a próxima camada. Este processo se repete até chegarmos na última camada.

Depois do forward propagation, vem o back propagation, que é o algoritmo responsável por ajustar os pesos e viéses da rede neural de modo que minimiza a nossa função de custo. Esse método é baseado no gradiente descendente e utiliza programação dinâmica e a regra da cadeia para calcular os gradientes de cada parâmetro no tocante à função de custo.

## Notação

Para realmente entender a teoria por trás das Redes Neurais nós precisamos primeiro introduzir uma notação, utilizaremos a notação do livro do Mostafa:

![](.\imagens\exemplo_rede_neural.png)

As camadas são rotuladas por $$l = 0,1,2, ..., L$$ e, como podemos ver, a camada $$l = 0$$ é a nossa camada de entrada - interessante dizer que, dependendo da fonte, a camada de entrada nem sempre é vista como uma camada por si só. Além disso, as camadas $$l = 1,2,...,L-1$$ são as chamadas camadas escondidas. Neste capítulo utilizaremos este $$\text{índice}^{(l)}$$ para fazer referência a uma camada em específico. Podemos dizer também que camadas possuem dimensões $$d^{(l)}$$: se uma camada $$d^{(l)}$$, isso significa que uma camada $$l=3$$ possui $$d(l)+1$$ nós que são rotulados $$0,1,...,d^{(l)}$$. Note que, na nossa representação, o nó 0 representa o viés daquele nó, e ele sempre possui como saída o valor 1, com nenhum valor de entrada no nó. 

Para expandir nossa notação, agora olharemos uma relação entre dois nós:

![](.\imagens\relacao_2_nos.png)

Observe que o nó $$j$$ possui um sinal $$s$$ de entrada com uma saída $$x$$, baseado nisso nós criaremos dois vetores: o vetor $$\bold{s}^{(l)}$$ e o vetor $$\bold{x}^{(l)}$$. O primeiro vetor - $$\bold{s}^{(l)}$$ - será o vetor sinal, ele representa os sinais de entrada recebidos pelos nós $$1,2,...,d^{(l)}$$ da camada $$l$$ - lembre-se que o nó zero não possui nenhum sinal de entrada. Já o segundo vetor - $$\bold{x}^{(l)}$$ - representa as saídas dos neurônios $$0,1,2,...,d^{(l)}$$ da camada $$l$$. Desse modo, a entrada $$s_{j}^{(l)}$$ é o sinal de entrada do nó $$j$$ da camada $$l$$, enquanto $$x_{j}^{(l)}$$ é a saída do nó $$j$$ da camada $$l$$.

Também precisamos de uma representação para os pesos. Dado que há conexões ligando as saídas de todos os nós da camada $$l-1$$ às entradas da camada $$l$$, podemos construir uma matriz de pesos $$W^{(l)}$$ de dimensões $$(d^{(l-1)} + 1) \times d^{(l)}$$. Além disso, cada elemento $$w_{ij}^{(l)}$$ da matriz $$W^{(l)}$$ é o peso que conecta o nó $$i$$ da camada $$l-1$$ ao nó $$j$$ da camada $$l$$. Portanto, nosso conjunto de matrizes $$\bold{w} = \{W^{(1)}, W^{(2)}, W^{(3)},...,W^{(L)}\}$$ reúne os parâmetros do modelo.

## Forward Propagation

Observe que, para conseguirmos construir o vetor de entradas da camada $$l$$, nós computamos a soma ponderada pelos pesos sobre as saídas da camada anterior. Ou seja, dados os pesos $$W^{(l)}$$, temos $$s_{j}^{(l)} = \sum_{i=0}^{d(l-1)}w_{ij}^{(l)}x_{i}^{(l-1)}$$. Este processo pode ser representado pela equação matricial:

$$
    \bold{s}^{(l)} = (W^{(l)})^{T}\bold{x}^{(l-1)}
$$

Computado o vetor $$s^{(l)}$$, podemos agora encontrar o vetor $$\bold{x}^{(l)}$$ dando o seguinte passo:

$$
    \bold{x} = 
        \begin{bmatrix}
            1 \\
            \theta(\bold{s}^{(l)}) 
        \end{bmatrix}
$$

Assim senedo, o algorítmo Forward Propagation pode ser representado pela cadeia de eventos:

$$
    \bold{x} = \bold{x}^{(0)}\xrightarrow{W^{(1)}}\bold{s}^{(1)}\xrightarrow{\theta}\bold{x}^{(1)}...\rightarrow\bold{s}^{(L)}\xrightarrow{\theta}\bold{x}^{(L)} = h(\bold{x})
$$

Com $$h(\bold{x})$$ sendo a saída do nosso modelo dada a entrada $$\bold{x}$$.

## Backpropagation

O algorítmo backpropagation é uma forma eficiente de se computar o gradiente $$\nabla\bold{\mathcal{L}(w)}$$ de uma função de perda que uma rede neural possui. Com frequência utilizamos o erro quadrático médio:

$$
    E_{in}(\bold{w}) = \frac{1}{N}\sum_{n=1}^{N}(h(\bold{x}_{n}^{(0)}) - y_{n})^{2}
$$

$$
   E_{in}(\bold{w}) = \frac{1}{N}\sum_{n=1}^{N}(\bold{x}_{n}^{(L)} - y_{n})^2
$$

$$
   E_{in}(\bold{w}) = \frac{1}{N}\sum_{n=1}^{N}e_{n}
$$

Para computar o gradiente de $$E_{in}$$, precisamos de suas derivadas parciais com respeito a cada matriz de pesos:

$$
    \frac{\partial\mathcal{L}(\bold{w})}{\partial W^{l}} = \frac{1}{N}\sum_{n=1}^{N}\frac{\partial e_{n}}{\partial W^{(l)}}
$$

Estas são as derivadas parciais que desejamos encontrar. Para encontrá-las, a ideia é computar as derivadas parciais da camada $$l$$ utilizando as derivadas parciais encontradas para a partir da camada $$l+1$$. Para fazer isso, precisamos introduzir $$\delta^{(l)}$$, chamado vetor de 'sensibilidade':

$$
    \delta^{(l)} = \frac{\partial e}{\partial \bold{s}^{(l)}} =
    \begin{bmatrix}
        \frac{\partial e}{\partial s_{1}^{(l)}} \\
        \frac{\partial e}{\partial s_{2}^{(l)}} \\
        ... \\
        \frac{\partial e}{\partial s_{d^{(l)}}^{(l)}}
    \end{bmatrix}
$$

A sensibilidade nos diz como $$e$$ muda dado $$\bold{s}^{(l)}$$. Podemos escrever:

$$
    \frac{\partial e}{\partial W^{(l)}} = \mathbf{x}^{(l-1)}(\delta^{(l)})^{T}
$$

A razão por trás disso será mostrada agora. Lembre-se da cadeia de eventos já introduzida para o forward propagation:

$$
    \bold{x}^{(0)}\xrightarrow{W^{(1)}}\bold{s}^{(1)}\xrightarrow{\theta}\bold{x}^{(1)}...\rightarrow\bold{s}^{(L)}\xrightarrow{\theta}\bold{x}^{(L)} = h(\bold{x})
$$

Note como o erro depende de $$s^{(l)}$$, que por sua vez depende de $$W^{(l)}$$ para ser encontrado. Observe que, devido à regra da cadeia, podemos escrever:

$$
    \frac{\partial e}{\partial w_{ij}^{(l)}} = \frac{\partial e}{\partial \bold{s}_{j}^{(l)}} . \frac{\partial \mathbf{s}_{j}^{(l)}}{\partial w_{ij}^{(l)}}
$$

Relembrando a definição do vetor $$s^{(l)}$$:

$$
    \bold{s}_{j}^{(l)} = \sum_{n=1}^{d^{(l-1)}}w_{nj}^{(l)}\bold{x}_{n}^{(l-1)}
$$

Temos, então:

$$
    \frac{\partial \bold{s}_{j}^{(l)}}{\partial w_{ij}^{(l)}} = \frac{\partial \sum_{n=1}^{d^{(l-1)}}w_{nj}^{(l)}\bold{x}_{n}^{(l-1)}}{\partial w_{ij}^{(l)}} = \bold{x}_{i}^{(l-1)}
$$

Lembre-se que $$\frac{\partial e}{\partial \bold{s}_{j}^{(l)}} = \bold{\delta}_{j}^{(l)}$$, e que $$\frac{\partial e}{\partial w_{ij}^{(l)}} = \frac{\partial e}{\partial \bold{s}_{j}^{(l)}} . \frac{\partial \bold{s}_{j}^{(l)}}{\partial w_{ij}^{(l)}}$$. Logo:

$$
    \frac{\partial e}{\partial w_{ij}^{(l)}} = \bold{x}_{i}^{(l-1)} . \bold{\delta}_{j}^{(l)}
$$

Desse modo, finalmente:

$$
    \frac{\partial e}{\partial W^{(l)}} = \bold{x}^{(l-1)}(\bold{\delta}^{(l)})^{T}
$$

A formula para os vetores de sensibilidade utilizada no backpropagation é:

$$
    \bold{\delta}^{(l)} = \sigma'(\bold{s}^{(l)}) \otimes [W^{(l+1)}\bold{\delta}^{(l+1)}]_{1}^{d^{(l)}}
$$

A $$\sigma'$$ é a derivada da função de ativação, o vetor $$[W^{(l+1)}\bold{\delta}^{(l+1)}]_{1}^{d^{(l)}}$$ contém os componentes do vetor $$W^{(l+1)}\bold{\delta}^{(l+1)}$$ (precisamos excluir o viés, que contém índice 0). O $$\otimes$$ é a notação para a multiplicação elemento a elemento, também conhecida como produto de Hadamard. 

Para entender a equação nós precisamos primeiro lembrar que:

$$
    e = e(\bold{x}^{(L)},y) = e(\sigma(\bold{s}^{(L)}),y)
$$

Devido à regra da cadeia, podemos escrever:

$$
    \frac{\partial e}{\partial \bold{s}^{(l)}} = \frac{\partial e}{\partial \bold{x}^{(l)}} . \frac{\partial \bold{x}^{(l)}}{\partial \bold{s}^{(l)}}
$$

$$
    \bold{\delta}_{j}^{(l)} = \frac{\partial e}{\partial\bold{s}_{j}^{(l)}} = \frac{\partial e}{\partial \bold{x}_{j}^{(l)}} . \frac{\partial \bold{x}_{j}^{(l)}}{\partial\bold{s}_{j}^{(l)}}
$$

Visto que $$\bold{x}^{(l)}$$ é uma função de $$\bold{s}^{(l)}$$, podemos escrever:

**(Comentário) Meninos, acho que teve um erro do Mostafa aqui. Ele escreve a seguinte sequência de equações:**

$$
    \frac{\partial \bold{x}_{j}^{(l)}}{\partial \bold{s}_{j}^{(l)}} = \sigma'(\bold{s}^{(l)}) \Longrightarrow \bold{\delta}_{j}^{(l)} = \frac{\partial e}{\partial \bold{x}_{j}^{(l)}} . \sigma'(\bold{s}^{(l)})
$$

**Creio que deveria ser:**
$$
    \frac{\partial \bold{x}_{j}^{(l)}}{\partial \bold{s}_{j}^{(l)}} = \sigma'(\bold{s}_{j}^{(l)}) \Longrightarrow \bold{\delta}_{j}^{(l)} = \frac{\partial e}{\partial \bold{x}_{j}^{(l)}} . \sigma'(\bold{s}_{j}^{(l)})
$$

**Me digam o que acham, por enquanto seguirei assumindo que o Mostafa estava correto**

Faz sentido dizer então que cada elemento de $$\bold{x}^{(l)}$$ influencia todos os elementos de $$s^{(l+1)}$$. Portanto, para obter a derivada que está faltando na equação acima, nós devemos levar em consideração todas essas influências:

$$
    \frac{\partial e}{\partial \bold{x}_{j}^{(l)}} = \sum_{k=1}^{d^{(l+1)}}\frac{\partial e}{\partial \bold{s}_{k}^{(l+1)}} . \frac{\partial \bold{s}_{k}^{(l+1)}}{\partial \bold{x}_{j}^{(l)}} = \sum_{k=1}^{d^{(l+1)}}\bold{\delta}_{k}^{(l+1)} . w_{jk}^{(l+1)}
$$

Finalmente, agora podemos encontrar a fórmula para as sensitividades:

$$
    \bold{\delta}_{j}^{(l)} = \sigma'(\bold{s}^{(l)}) . \sum_{k=1}^{d^{(l+1)}}\bold{\delta}_{k}^{(l+1)} . w_{jk}^{(l+1)}
$$

Agora nós encontramos uma forma de computar $$\delta^{(l)}$$ a partir de $$\delta^{(l+1)}$$. Isso significa que, para encontrar todas as sensitividades, basta encontrarmos primeiro $$\delta^{(L)}$$:

$$
    \bold{\delta}^{(L)} = \frac{\partial e}{\partial \bold{s}^{(L)}} = \frac{e(\bold{x}^{(L)}, y)}{\partial \bold{s}^{(L)}}
$$

Isso significa que $$\delta^{L}$$ é dependente da função de custo que a rede neural deseja minimizar. Tendo como exemplo a soma das diferença dos quadrados, isso é: $$e = (\bold{x}^{(L)} - y)^{2} = (\sigma(\bold{s}^{(L)}) - y)^{2}$$, temos:

**Essas derivadas com vetores aqui tão certas?**

$$
    \bold{\delta}^{(L)} = \frac{\partial e}{\partial \bold{s}^{(L)}}
$$

$$
    \bold{\delta}^{(L)} = \frac{\partial(\mathbf{x}^{(L)} - y)^{2}}{\partial\mathbf{s}^{(L)}}
$$

$$
    \bold{\delta}^{(L)}= 2(\mathbf{x}^{(L)} - y)\frac{\partial\mathbf{x}^{(L)}}{\partial\mathbf{s}^{(L)}}
$$
$$
    \bold{\delta}^{(L)} = 2(\mathbf{x}^{(L)} - y)\sigma'(\mathbf{s}^{(L)})
$$
