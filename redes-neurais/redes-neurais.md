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