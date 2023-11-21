# Self-Attention

Como dito anteriormente, o mecanismo de "Self-Attention" é um dos protagonistas do modelo de Transformers. 

Para entender esse conceito, é importante saber o significado de "Embbeding". Quando estamos falando sobre aplicações de machine learning que envolvem palavras (evidentemente o caso de NMT).

Nesse sentido, para que possamos trabalhar com palavras de um modo que seja adequado para modelos de Machine Learning, precisamos representá-las de outra maneira. Assim, transformamos cada palavra em um vetores em um espaço n-dimensional. Isso é feito pelo processo de embbeding. A forma mais imediata de se fazer essa vetorização de palavras é por meio do, já conhecido, "One-Hot Encoding".

Nesse método, o alfabeto (todas as palavras contempladas pelo modelo) de tamanho $$n$$ é a base canônica do espaço n-dimensional, em que cada palavra é um vetor da base canônica.

Note que esse método não é tão representativo do sigificado semântico das palavras. Por exemplo, se considerarmos as palavras "visão", "olho" e "diamante" é evidente que as palavras "visão" e "olho" tem significado mais próximo que "diamante". Contudo, para o computador, após realizar "One-Hot Enconding" essas palavras estarão igualmente distantes. Então, fica claro que essa maneira de representação de palavras não é tão boa.

Assim, o mecanismo de Self-Attention, se propõe a produzir uma representação de palavras (embbeding) mais fidedigna ao real significado das palavras.
Esse novo embbeding se propõe a fazer isso ao considerar o contexto da frase para ser produzido. Então, caso, em uma frase, as palavras "visão" e "olho" tenham significado próximo ao se levar em conta o contexto, o embbeding oriundo de Self-Attention será feito de tal modo que essas palavras estejam próximas no espaço n-dimensional.

Assim, para fazer esse embbeding para uma palavra específica em uma frase, são associados pesos à todas as palavras dessa frase. Esses pesos são tais que as palavras vizinhas com maiores pesos têm maior semelhança contextual em relação à palavra sendo analisada. 

Agora, como podemos obter esses pesos que são associados às palavras em uma frase?
O primeiro passo é definir um vetor $$s_{ij}$$ tal que:
$$
s_{ij} = v_i \cdot v_j
$$
Isso é o produto escalar entre as palavras $$v_i$$ e $$v_j$$. Note que, por definição,
$$
v_i \cdot v_j = \|v_i\|\|v_j\|\cos{\theta}
$$

Interpretando as palavras $$v_i$$ e $$v_j$$ como vetores no espaço n-dimensional, perceba que o valor do cosseno será maior para $$\theta$$ pequeno. Então, esse produto escalar nos dará valores altos para palavras cujos embeddings (pontos) estão próximos no espaço n-dimensional.

<div align="center">
<img src="images/fig1.png" alt="Obtendo scores via dot product" width="300"/>
</div>


Depois disso, normalizamos os valores desses scores com Softmax.



{% hint style="info" %}
Note que esse peocesso é muito semelhante ao processo de obtenção dos pesos $$\alpha_{ij}$$ em Attention. A diferença é que, naquele caso, fazíamos score de semelhança entre uma palavra do output para cada vetor de contexto do input. Já aqui, fazemos isso entre as palavras do input (no encoder).
{% endhint %}

## Normalizando
$$
w_{ij} = \text{softmax}(s_{ij}) = \dfrac{e^{s_{ij}}}{\sum^n_{k=1}s_{ik}}
$$
Aqui, cada vetor de score $$(s_{i1}, \ldots, s_{in})$$ estará normalizado tal que seus correspondetes $$(w_{i1}, \ldots, w_{in})$$ são tais que:$$\sum^n_{j=1}w_{ij} = 1$$.

<div align="center">
<img src="images/fig2.png" alt="Normalizando os scores" width="300"/>
</div>

Agora, nós vamos utilizar esses vetores de pesos $$(w_{ij})$$ como segue:
$$
y_i = \sum^n_{j=1}w_{ij} \cdot v_j
$$

Note que, dessa maneira, teremos $$n$$ vetores $$y_i$$ que são somas dos vetores das palavras multiplicadas pelos pesos $$w_{ij}$$.

{% hint style="info" %}
Por isso, o vetor $$y_i$$ será mais influenciado pelas palavras $$v_j$$ tais que $$w_{ij}$$ é maior. Isto é, palavras $$j$$ com maior relação com a palavra $$i$$.
{% endhint %}

$$y_i$$ 
é a forma contextualizada do vetor $$v_i$$.


{% hint style="info" %}
Nesse ponto, há considerações importantes.

Primeiramente, note que estamos usando os mesmos vetores de palavras $$v_i$$ em três papeis distintos:

* 1º e 2º: usamos $$\underline{v_i} \cdot \underline{v_j}, \; j = 1, \ldots, n$$ para obter scores.
* 3º: usamos $$w_{ij} \cdot \underline{v_j}$$ para obter $$y_i$$

Assim, esses vetores estão sendo colocados de maneira crua nessas operações, que buscam obter o vetor $$y_i$$ contextualizado.

Isso não faz sentido, pois somente o embedding cru (usado nos vetores de palavras e no dot product) não será suficiente para explorar o contexto da frase.

Imagine que, se isso acontecesse, no espaço n-dimensional poderia ocorrer de: 
$$
\|v_1\| << \|v_2\|
$$

Assim, se computássemos $$y_1$$ daquela maneira ingênua, $$y_1$$ poderia acabar sofrendo uma influência indevida de $$v_2$$ somente por sua norma ocasionada pelo embedding ser muito grande.
{% endhint %}

Para abordar essa questão, inserimos matrizes de pesos treináveis ($$\text{Q, K, V}$$) para reger a influência dos vetores de palavras $$v_i$$ no 1º, 2º e 3º estágio de obtenção dos vetores contextualizados $$y$$.

{% hint style="warning" %}
Note que, como essas matrizes serão obtidas treinando o modelo focando na captura de informação sobre contexto, elas serão capazes de explorar as características de contexto melhor que o embedding cru. Isso faz sentido?}
{% endhint %}

<div align="center">
<img src="images/fig3.png" alt="Interação entre $$\text{Q, K, V}$$" width="300"/>
</div>


Perceba que a intuição de que as matrizes $$\text{Q, K, V}$$ serão capazes de capturar inteiramente as características de contexto pode parecer frágil. Ainda pensando que existem diversos contextos diferentes (semânticos) em uma mesma frase. Essas matrizes poderiam se especializar em representar o contexto de forma análoga ao embedding... (Afinal, de certa forma embedding também considera contexto).

Daí surge o conceito de multi-headed Attention. Introduzimos diversas matrizes $$\text{Q, K, V}$$. Isto é, várias camadas da figura 1. Desse modo, teremos, ao final do processo, um embedding contextualizado para cada camada. A ideia é que cada um desses vetores $$y$$ seja uma representação contextual diferente da palavra em relação à frase.

Então, como queremos somente um vetor que contenha toda essa informação, concatenaremos esses vetores $$y$$ e depois o fazemos passar por uma matriz de pesos W que fará esse vetor concatenado ter a dimensão (tamanho) desejado para y.

Nesse ponto, podemos introduzir a noção matricial que multiheaded self attention possui. Perceba que essa noção é matricial mas ainda estamos trabalhando com um exemplo de um input individual (sem batches de dados). Depois de passarmos por todo o mecanismo de self attention, podemos introduzir batches no raciocínio.


{% hint style="warning" %}
Como funciona o treinamento para ajuste de pesos $$\text{Q, K, V}$$ e $$W$$?

Acho que precisamos chegar ao final do decoder para começar a visualizar isso
{% endhint %}


[^1]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need.
