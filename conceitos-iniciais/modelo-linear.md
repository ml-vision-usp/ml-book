# Modelo linear



Agora que já temos definido como seria a entrada dessa função ($\textbf{x}\_i)$ e sua saída ($\hat{y}\_i$), precisamos pensar como seria a função propriamente dita. Primeiramente, podemos elencar dois tipos principais de problemas: os de classificação e os de regressão.

## 1 - Classificação:

Os problemas de classificação podem ser interpretados como problemas em que, dado uma entrada $\textbf{x}\_i$, o valor que iremos prever ($y\_i$) --- perceba que iremos prever um valor de $y\_i$, mas o valor que iremos obter efetivamente é $\hat{y}\_i$ --- simboliza uma variável discreta. Repare que no caso de prever se gostaremos ou não de um filme, a variável $y\_i$ é binária, e só poderia assumir, por exemplo, os valores $1$ ou $0$. Em outra situação, de diferenciar cachorros de gatos, poderíamos utilizar $y\_i = 0$ para representar gatos e $y\_i = 1$ cachorros. \\

De qualquer forma, quando estamos tratando de problemas de classificação, alguns deles podem ser caracterizados como linearmente separáveis. Isso quer dizer que, a partir de todas as entradas $\textbf{x}\_i$ já observadas --- que utilizaremos para treinar a máquina --- há uma estrutura linear (hiperplano) que separa os dados em suas devidas classes. Conseguiremos visualizar essa característica através da implementação efetuada na seção sobre Regressão Logística, em que decidiremos se um cliente $i$ comprou ($y\_i = 1$) ou não ($y\_i = 0$) um determinado produto.

## 2 - Regressão:

Por outro lado, nos problemas de regressão, apesar de termos a mesma entrada $\textbf{x}\_i$, $y\_i$ será uma variável contínua. No lugar de atribuir à entrada alguma classe, nós obtemos um valor contínuo. Por exemplo, dado o histórico escolar, estimar o salário futuro de estudantes universitários ou um banco decidir a quantidade ideal de crédito que se deve fornecer para um dado cliente. Em vista disso, há situações em que o valor da variável $y\_i$ varia de maneira consideravelmente linear de acordo com a entrada $\textbf{x}\_i$.





Após considerarmos os problemas que podem ser resolvidos de maneira direta usando modelos lineares, é natural surgir indagações sobre quais ferramentas utilizaríamos caso esse caráter de certa linearidade nos dados não esteja presente. Nesse sentido, para resolver questões que não apresentam tais características de linearidade, o procedimento aplicado é o de combinar, diversas abordagens lineares --- injetando não linearidade entre essas abordagens, de uma maneira que será explicada na seção sobre Redes Neurais --- e, dessa forma, conseguir resolver o problema. Estudaremos mais esse tópico na seção sobre Redes Neurais. \\

Assim, como todas as técnicas de aprendizado de máquina cobertas nesse texto se apoiam em pilares de linearidade, definiremos o Modelo Linear, que estará fortemente presente de agora em diante. \\

Desse modo, unindo as ideias apresentadas até aqui, a ideia central dos algoritmos abordados ao longo das próximas seções é representar os problemas que desejamos solucionar como funções que possuem uma essência, de certa forma, linear. Contudo, apesar de termos conhecimento sobre o formato dessa função, não a conhecemos de fato. \\

Nesse sentido, para adquirir uma melhor noção sobre essa função $h$ que queremos construir para aproximar $f$, é válido pensar na representação de dados como exemplos:

$$
(\textbf{x}_1, y_1) , (\textbf{x}_2, y_2), \ldots, (\textbf{x}_N, y_N)
$$

Examinando mais a fundo um vetor $\mathbf{x}\_i$ que esteja dentro desse conjunto de exemplos, temos o que segue:

$$
\mathbf{x}_i = \begin{bmatrix} x_{i,1} \\ x_{i,2} \\ \vdots \\ x_{i,d} \end{bmatrix}
$$

Os elementos $x\_k, ; k = 1, \ldots, d$ desse vetor são chamados de características (\emph{features}) da entrada e $d$ é a dimensão dos vetores de entrada $\mathbf{x}$. Voltando ao exemplo dos filmes, e considerando $d = 3$ essas features podem ser tais que: \begin{itemize} \item $x\_{i,1}$ representa a quantidade de ação no filme $i$ \item $x\_{i,2}$ representa a quantidade de aventura no filme $i$ \item $x\_{i,3}$ representa a quantidade de drama no filme $i$ \end{itemize}

Agora, sabendo que uma função linear é da forma:

$$
h(x_1,x_2,\ldots,x_d) = a_0 + a_1x_1 + a_2x_2 + \ldots + a_dx_d
$$

Vamos considerar um vetor $\mathbf{w}$ (o qual chamaremos de vetor de pesos) tal que:

$$$
\begin{bmatrix} w_0 \\ w_1 \\ \vdots \\ w_d \end{bmatrix}$$ Dessa forma, queremos relacionar os pesos $w_k, \; k = 0,\ldots, d$ com as features $x_{i,k}, \; k = 1,\ldots,d$ de cada exemplo $\mathbf{x}_i$ de maneira linear tentando obter uma expressão que aproxime o comportamento da função $f$ que leva $\mathbf{x}_i$ a $y_i$, com $i = 1, \ldots, N$. \\ A fim de realizar tal tarefa, vamos relacionar $\mathbf{x}_i$ e $\mathbf{w}$ da seguinte maneira: $$h(\mathbf{x}_i) = w_0 + w_1x_{i,1}+ w_2x_{i,2} + \ldots + w_dx_{i,d}$$ Para facilitar a notação, podemos inserir o elemento $x_{i,0} = 1$ em cada vetor de exemplo $\mathbf{x}_i$. Denotaremos $\mathbf{x}_i$ adicionado desse elemento por $\mathbf{\tilde{x}}_i$: $$h(\mathbf{x}_i) = w_0 + w_1x_{i,1}+ w_2x_{i,2} + \ldots + w_dx_{i,d} = \mathbf{w}^T\mathbf{\tilde{x}}_i$$ Esse é o modelo linear. Por fim, o processo por trás da abstração de aprendizado envolvida em Machine Learning consiste em utilizar os exemplos de dados $(\mathbf{x}_i, y_i)$ para produzirmos $h$ que aproxime $f$. Isso será feito encontrando o vetor de pesos $\mathbf{w}$ que melhor faça esse papel. A maneira de encontrar esse valor ótimo para $\mathbf{w}$ é fundamentada pela chamada ``Função de Perda'', que explicaremos a seguir.
$$$
