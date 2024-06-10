# Redes Neurais Recorrentes

## Introduzindo o Problema

Imagine que você deseja resolver o seguinte problema: 

<p style="text-align: center;"><em><b>Crie um modelo capaz de copiar as obras de William Shakespeare, escrevendo de forma semelhante ao renomado escritor.</b></em></p>

Como você poderia possívelmente modelar esse problema para um modelo de *Machine Learning*? Redes Neurais podem ser utilizadas para resolver esse problema ou seria necessário um modelo diferente para resolvê-lo?

## Modelando Shakespeare

Como estamos trabalhando com modelos numéricos, precisamos pensar em uma forma de modelar palavras de forma numérica. Para nossa exploração, vamos tentar transformar em dados numéricos a seguinte frase:

<p style="text-align: center;"><em> "Better three hours too soon than a minute too late"</em></p>

Como primeira ideia, poderíamos definir cada palavra como um número, transformando a palavra *"Better"* no número 1, *"three"* no número 2 e assim successivamente. Dessa forma, nossa frase poderia ser representada pelo vetor [1, 2, 3, 4, 5, 6, 7, 8, 4, 9]. Inicialmente, pode parecer interessante, porém introduz em modelos numéricos como redes neurais um viés indesejado: Palavras com representadas com números próximos acabam sendo vistas como próximas em significado. Por exemplo, as palavras "Better" (índice 1) e "hours" (índice 3) seriam vistos como mais próximas em significado do que as palavras "hours" e "minute" (índice 8), o que não faz muito sentido. 

Para resolver esse problema, podemos aplicar uma técnica chamada de **One-Hot Encoding**, que evita que essas falsas associações  sejam definidas pelo modelo. Basicamente, a ideia é definir um vetor do tamanho do seu vocabulário, que no nosso caso é de apenas 9 palavras e, para cada i-ésima palavra , definir a i-ésima entrada do vetor como um e zero para as restantes. Assim, para a palavra "Better", teríamos o vetor [1, 0, 0, 0, 0, 0, 0, 0, 0] e para a palavra "too" teríamos o vetor [0, 0, 0, 1, 0, 0, 0, 0, 0].



## Recursos Úteis

- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)