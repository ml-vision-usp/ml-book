# Gradiente Descendente

A ideia desse algoritmo de otimização é, dado uma função côncava, para encontrar o ponto de mínimo, basta `caminhar'' sempre na direção de variação mais negativa da função a partir do ponto atual. Como será explicado, essa direção é a direção do vetor oposto ao gradiente da função no ponto. Com isso, teremos o seguinte algoritmo de atualização do vetor de pesos {\bf w}: \begin{gather*} {\bf w}(t+1) \leftarrow {\bf w}(t) - \eta\dfrac{\nabla l({\bf w})}{\lVert \nabla l({\bf w}) \rVert} \end{gather*} Onde $\eta$ é a chamada` taxa de aprendizado''.
