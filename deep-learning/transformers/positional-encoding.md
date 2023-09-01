# Positional Encoding

Lembrando das RNNs, havia (de modo inerente ao modelo) uma ideia de sequencialidade entre os inputs (palavras).

Isso porque, tanto no encoder (vetor de anotação, vetor de contexto, hidden state) quanto no decoder (palavra, prevista, hidden state) as informações obtidas sobre a palavra $$i$$ levavam em consideração informações sobre as palavras anteriores.

Até mesmo pelo fato do input ser sequencial.

{% hint style="info" %}
Questão sobre paralelismo em RNNs e Transformers
{% endhint %}

Agora, no caso de Transformers, essa ideia de sequencialidade nas palavras de input não são inerentes à arquitetura do modelo. → Reflita sobre como o mecanismo de Self-Attention funciona sem ideia de sequencialidade.

Apesar disso, é evidente que a posição das palavras em uma frase tem muita importância na interpretação do seu significado naquela frase.

Assim, Positional Encoding é uma maneira de embutir essas informações sobre localidade no embedding das palavras.

Esse será o embedding que é forncecido como $$v_i$$ para o encoder.

De maneira geral, somamos ao embedding de uma palavra um vetor que tenha a informação sobre a posição da palavra na frase. 