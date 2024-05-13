# Regressão Logística

A regressão logística é um método que dialóga muito com problemas de classificação. Assim como a regressão linear, a regressão logística tem como saída um valor real. Todavia, esse valor real é limitado no intervalo [0,1], de modo que a saída representa uma probabilidade.

## Ideia Principal

A regressão logística é um modelo linear, o que nos possibilita dizer que, assim como a regressão linear, ela é baseada em um sinal $$s = \bold{w}^{T}\bold{x}$$, com $$\bold{w}$$ sendo um vetor coluna de pesos e $$\bold{x}$$ sendo um vetor coluna que representa a entreda do nosso modelo. Nesse sentido, para que consigamos transformar esse sinal $$s$$ em um valor entre [0,1], uma forma possível é utilizar uma função logística $$\theta(s)$$ cujos resultados são limitados entre 0 e 1 e é definida como:

$$
    \theta(s) = \frac{e^s}{1+e^s}
$$

Essa função é muito interessante e estudaremos ela para problemas de saída binária (por exemplo, se um banco deve ou não aprovar crédito para uma pessoa com base nos seus dados bancários). Note que o que nós queremos que nosso modelo faça é prever a probabilidade de que uma saída $$y_i$$ seja 1 ou -1, por exemplo, dado $$\bold{x}_i$$. Ou seja, queremos:

$$
    P(y_i|\bold{x}_i) = \theta(\bold{w}^T\bold{x}_i)
$$ 

Assumindo que nosso conjunto de dados de entrada $$X$$ representa a distribuição de probabilidade real, parece justo lidar com o conceito de verossimilhança para avaliar a nossa função $$\theta$$. A verossimilhança mede o quão provável é que o nosso conjunto de dados de aprendizado tenha sido visto, assumindo que os pontos $$(x_1,y_1), (x_2,y_2), ..., (x_n,y_n)$$ foram gerados de maneira independente, podemos escrever a verossimilhança como sendo o produto:

$$
    \prod_{i=1}^{i=N} P(y_i|\bold{x}_i
$$

Como nós queremos que nosso modelo se aproxime da probabilidade real, isso significa que estamos procurando $$\bold{w}$$ que maximize o produto:

$$
    \prod_{i=1}^{i=N} \theta(\bold{w}^{T}\bold{x}_i)
$$

Ou, que minimize o somatório:

$$
    \frac{1}{N}\sum_{i = 1}^{N}ln(\frac{1}{\theta(\bold{w}^{T}\bold{x})})
$$

Nossa função de erro então é:

$$
    Erro(\bold{w}) = \frac{1}{N} \sum_{i=1}^{N}ln(1 + e^{-\bold{w}^{T}\bold{x}}) 
$$

