# Decoder da arquitetura Transformer

Agora que já sabemos como o decoder de um modelo transformer funciona, precisamos entender como vamos do embedding contextualizado entregue pelo encoder até uma frase totalmente traduzida (no contexto de tradução). Para entender esse processo, precisamos cobrir o funcionamento do decoder do modelo transformer.

Nesse sentido, a partir de uma perspectiva de muito alto nível, o encoder e o decoder interagem da seguinte maneira:

<div align="center">
<img src="images/fig4.png" alt="Alt text for the image" width="300"/>
</div>

Quando estamos falando de Transformers Decoder, temos a seguinte figura esquemática:

<div align="center">
<img src="images/fig5.png" alt="Alt text for the image" width="300"/>
</div>

É importante que façamos uma distinção entre o funcionamento do decoder em estágio de treinamento e inferência. Uma sacada muito interessante implementada na arquitetura transformer é modelar uma tarefa inerentemente sequencial (prever a próxima palavra, dado as palavras anteriores em um texto) de modo que, no momento de treinar a rede neural para efetuar essa tarefa, esse treinamento possa ser feito de maneira paralela. 

Contudo, em momento de inferência, a tarefa certamente precisa ser feita de modo sequencial. Assim, para abordar essa questão, primeiramente trataremos do período de treinamento.

## Decoder em Estágio de Treinamento
Nessa situação, podemos pensar que o modelo recebe como inputs os seguintes dados:

1. Frase inteiramente traduzida
2. embeddings contextualizados das palavras no idioma original.

Em um primeiro momento de atuação do Decoder, a frase inteiramente traduzida é submetida à Masked Self Attention.

### Masked Self-Attention

Aqui, a ideia é estabelecer a relação do próximo output (próxima palavra predita) com os outputs anteriores.

{% hint style="warning" %}
Podemos pensar em uma analogia com os hidden states do decoder de uma RNN
{% endhint %}

Para entender como Masked Self-Attention funciona, podemos pensar na matriz de Self-Attention:

Dado as palavras: $$t_1, t_2  \text{ e } t_3$$ ($$t$$ de "traduzido"), temos a seguinte matriz:

<div align="center">
<img src="images/masked0.png" alt="Alt text for the image" width="300"/>
</div>

Onde o elemento $$ij$$ denota a intensidade da relação contextual entre as palavras $$t_i$$ e $$t_j$$.

Agora, perceba que, em momento de treinamento, como já possuímos a frase traduzida inteira ($$t_1, t_2 \text{ e } t_3$$), essa matriz pode ser inteiramente preenchida no momento inicial. Ou seja, podemos traçar as relações contextuais: $$t_1t_1, t_1t_2, t_1t_3$$ mesmo que, na prática, em momento de inferência, isso não seria possível (já que ainda não existiria $$t_2$$ ou $$t_3$$).

Nesse ponto, entra o conceito de Masked Self-Attention A ideia aqui é transformar a matriz de Sel-Attention (ilustrada anteriormente) em algo do tipo:

<div align="center">
<img src="images/masked1.png" alt="Alt text for the image" width="300"/>
</div>

{% hint style="info" %}
Após Softmax, cada $$-\infty$$ se torna 0
{% endhint %}


Desse modo, perceba que, apesar da frase inteira estar disponível, a matriz de Self-Attention resultante estará simulando o que ocorre em tempo de inferência.

Ou seja, $$t_1$$ só pode ter relação contextual analisada com $$t_1$$. → Somente $$t_1t_1$$.

$t_2$ pode ter relação com $$t_1$$ e $$t_2$$. ⇒ $$t_2t_1;t_2t_2$$.

E assim por diante. Perceba que dessa maneira podemos dar a frase inteira como input simulando uma geração sequencial das palavras.

Esse estágio de Self-Attention no Decoder é muito importante pois a próxima palavra traduzida prevista não está somente em função do output do Encoder. Essa palavra deve estar em função de todas as palavras anteriormente traduzidas.

### Encoder-Decoder Attention
Agora, depois de passarmos por Masked-Self Attention, com o que estamos lidando?

Primeiro obtemos a matriz de self-attention e a utilizamos para obter o embedding contextualizado de uma palava traduzida em relação à própria frase traduzida. Assim, no momento de treinamento, cada palavra da frase traduzida origina um vetor $$y$$ (que é o embedding contextualizado):

<div align="center">
<img src="images/fig6.png" alt="Alt text for the image" width="300"/>
</div>



Depois disso, chegamos à etapa de Attention entre Encoder e Decoder. Nessa etapa usaremos o segundo tipo de dado que o decoder recebe como entrada (embeddings contextualizados da frase no idioma original).

Esse conceito de Attention é muito semelhante ao proposto por Attention[^2], no qual, quantificamos a relação contextual da próxima palavra predita com todas as palavras da frase no idioma original.

Perceba que, se formos utilizar a analogia de $Q$, $K$, $V$ nesse caso de Encoder-Decoder Attention, poderemos pensar em um esquema como o seguinte:

<div align="center">
<img src="images/fig7.png" alt="Alt text for the image" width="300"/>
</div>

Cada vetor contextualizado resultado do Encoder é relacionado com cada vetor contextualizado da palavra sendo predita. Esse relacionamento é treinável (devido a matriz de pesos $K_{\text{encdec}}$). Obtemos scores dessa relação. A palavra mais relacionada terá maior score e, então, influenciará mais o output.

Então, normalizamos os scores e os utilizamos para ponderar uma nova representação das palavras do idioma original. Essa nova representação é originada pela matriz de pesos treináveis ($V_{\text{encdec}}$).

Agora, cada palavra é um vetor de tamanho $d$ (lembrando que $d$ é o tamanho do embedding) e precisamos ir transformar esse vetor em um vetor de tamanho $N$, onde $N$ é o tamanho do vocabulário e o valor do elemento $i$ é um score que está relacionado à quão provavel é que a próxima palavra predita seja a palavra $i$ do vocabulário. Note que, caso o embedding das palavras de entrada forem One-Hot encoding, temos que $d = N$. De qualquer forma, para obter esse vetor, basta uma camada linear -- que é mostrada na figura.

Então, para quue esses valores de quão provável a palavra $i$ do vocabulário seja a próxima palavra predita se tornem, de fato, probabilidades, utilizamos a função softmax.

Agora, temos um vetor do tamanho do vocabulário em que cada elemento $i$ corresponde à probabilidade de escolher a palavra $i$ como a próxima palavra da frase traduzida. Assim, para que haja variabilidade nas traduções, é possível que, no lugar de somente tomar $\text{argmax}$ desse vetor, amostremos $i$ de acordo com as probabilidades do vetor. 

### Considerações Sobre Treinamento
É razoável pensar tradução como um problema de classificação. Portanto, podemos usar Cross-Entropy como função de perda.

{% hint style="info" %}
Em momento de treinamento, o erro de uma predição não é propagado para a predição das próximas palavras traduzidas.


Assim, de fato usamos como ``palavras previamente traduzidas'' as palavras corretamente traduzidas. Esse conceito é conhecido como ``teacher forcing''.

Isso faz com que o processo de treinamento de uma rede Transformer seja totalmente paralelizável.

{% endhint %}


## Decoder em Estágio de Treinamento

Agora, quanto ao decoder em fase de inferencia, acontece um processo muito semelhante ao descrito. A única diferença é que o processo não é paralelizável e, por isso, acontece até de maneira mais intuitiva. Uma palavra é predita de cada vez, levando em consideração as palavras preditas anteriormente e o contexto da frase no idioma original. Perceba que, para isso acontecer, não precisamos mudar nada na arquitetura do modelo, até mesmo a camada de Masked Self-Attention, se mantém a mesma.


[^1]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need.
[^2]: Bahdanau, D., Cho, K., and Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate.

