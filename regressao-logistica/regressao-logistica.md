# Regressão Logística

Antes de começar a falar sobre nosso modelo de fato, é interessante primeiro explicar o que é um problema de classificação. Assim sendo, um problema de classificação é um desafio o qual, dado uma entrada, devemos inferir a sua saída. Todavia, diferente da regressão, na classificação temos um conjunto finito de saídas possíveis. Por exemplo: imagine que você é o gerente de um banco e, dado um conjunto de informações sobre a vida financeira de uma pessoa, você deve decidir se o banco deve ou não conceder um empréstimo para essa pessoa. Perceba que nesse problema a saída é binária e, portanto, finita: ou você concede o empréstimo, ou não concede o empréstimo.

A regressão logística é um método que dialóga muito com problemas de classificação, apesar de ser um modelo de regressão. Assim como a regressão linear, a regressão logística tem como saída um valor real. Todavia, esse valor real é limitado no intervalo [0,1], de modo que a saída representa uma probabilidade. Assim sendo, podemos utilizar esse modelo para inferir a probabilidade de uma saída de um problema de classificação.

## Ideia Principal

A regressão logística é um modelo linear, o que nos possibilita dizer que, assim como a regressão linear, ela é baseada em um sinal $$s = \bold{w}^{T}\bold{x}$$, com $$\bold{w}$$ sendo um vetor coluna de pesos e $$\bold{x}$$ sendo um vetor coluna que representa a entreda do nosso modelo. Nesse sentido, para que consigamos transformar esse sinal $$s$$ em um valor entre [0,1], uma forma possível é utilizar uma função logística $$\theta(s)$$ cujos resultados são limitados entre 0 e 1 e é definida como:

$$
    \theta(s) = \frac{e^s}{1+e^s}
$$

Essa função é muito interessante e estudaremos ela para problemas de saída binária (por exemplo, se um banco deve ou não aprovar crédito para uma pessoa com base nos seus dados bancários). Note que o que nós queremos que nosso modelo faça é prever a probabilidade de que uma saída $$y_i$$ seja 1 ou -1, por exemplo, dado $$\bold{x}_i$$. Ou seja, levando em consideração que $$1-\theta(s) = \theta(-s)$$, queremos:

$$
    P(y_i|\bold{x}_i) = \theta(y_i\bold{w}^T\bold{x}_i)
$$ 

Assumindo que nosso conjunto de dados de entrada $$X$$ representa a distribuição de probabilidade real, parece justo lidar com o conceito de verossimilhança para avaliar a nossa função $$\theta$$. A verossimilhança mede o quão provável é que o nosso conjunto de dados de aprendizado tenha sido visto, assumindo que os pontos $$(x_1,y_1), (x_2,y_2), ..., (x_n,y_n)$$ foram gerados de maneira independente, podemos escrever a verossimilhança como sendo o produto:

$$
    \prod_{i=1}^{i=N} P(y_i|\bold{x}_i
$$

Como nós queremos que nosso modelo se aproxime da probabilidade real, isso significa que estamos procurando $$\bold{w}$$ que maximize o produto:

$$
    \prod_{i=1}^{i=N} \theta(y_i\bold{w}^{T}\bold{x}_i)
$$

Ou, que minimize o somatório:

$$
    \frac{1}{N}\sum_{i = 1}^{N}ln(\frac{1}{y_i\theta(\bold{w}^{T}\bold{x}_i)})
$$

Nossa função de erro então é:

$$
    Erro(\bold{w}) = \frac{1}{N} \sum_{i=1}^{N}ln(1 + e^{-y_i\bold{w}^{T}\bold{x}}) 
$$

## Minimizando o erro

Assim como fizemos na regressão linear, a minimização da nossa função de erro será feita utilizando o gradiente descendente. O primeiro passo a ser dado é encontrar o vetor gradiente da nossa função de erro em relação ao vetor de pesos $$\bold{w}$$, o qual é:

$$
    \nabla Erro(\bold{w}) = -\frac{1}{N}\sum_{i=1}^{N}\frac{y_i\bold{x}_i}{1+e^{y_i\bold{w}^T\bold{x}_i}}
$$
$$
    \nabla Erro(\bold{w}) = \frac{1}{N}\sum_{i=1}^{N}-y_i\bold{x}_i\theta(-y_i\bold{w}^T\bold{x}_i))
$$

Desse modo, podemos seguir com o algoritmo do gradiente descendente e assim encontramos nossos pesos.