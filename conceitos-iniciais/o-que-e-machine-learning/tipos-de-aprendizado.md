# Os Diferentes Tipos de Aprendizado

O mundo real é cheio de problemas. Explorá-los não é fácil e muitas vezes precisaremos de soluções inteligentes para resolvê-los. Podemos utilizar diferente tipos de aprendizado para resolver diferentes modalidades de problema na área de *Machine Learning*. Hoje vamos apresentar tais tipos de aprendizado, aplicações comuns desses métodos e vantagens e desvantagens de diferentes abordagens. Abaixo, temos os três tipos de aprendizado mais comuns na indústria:

### 1. Aprendizado Supervisionado
#### Definição
No aprendizado supervisionado, o modelo é treinado com um conjunto de dados rotulados, onde cada exemplo de treinamento é composto por uma entrada e a resposta desejada (saída). O objetivo é aprender uma função que mapeia entradas para saídas.

#### Técnicas Comuns

- **Regressão Linear**: O modelo aproxima uma função linear à partir de um conjunto de dados. Podemos utilizar para previsão de preço de aluguel de uma região à partir de renda média local.

- **Regressão Logística**: Modelo que aprende a classificar dados linearmente separáveis. Por exemplo, diferenciar maçãs de bananas pela cor.

- **Redes Neurais**: Modelo que tem potencial modelar qualquer função contínua, desde que tenha dados suficientes e arquitetura adequada. É amplamente usada hoje em modelos como ChatGPT e Stable Diffusion.

#### Vantagens e Desvantagens
**Vantagens**: Geralmente oferece alta precisão, é fácil de interpretar e validar, especialmente em problemas de classificação e regressão.
**Desvantagens**: Requer grandes quantidades de dados rotulados, o que pode ser caro e demorado para obter.

### 2. Aprendizado Não Supervisionado
#### Definição
No aprendizado não supervisionado, o modelo é treinado com dados que não possuem rótulos. O objetivo é encontrar padrões ou estruturas subjacentes nos dados.

#### Técnicas Comuns

- **Clusterização**: Agrupa os dados em clusters baseados em similaridades (ou seja, grupos de pontos próximos). Exemplos incluem a segmentação de clientes em marketing ou a identificação de padrões em dados de saúde.

- **Redução de Dimensionalidade**: Reduz o número de variáveis em um conjunto de dados, mantendo o máximo de informações possível. Técnicas como PCA (Análise de Componentes Principais) são amplamente utilizadas.

#### Vantagens e Desvantagens
- **Vantagens**: Pode revelar padrões ocultos nos dados, útil para explorar e entender dados sem rótulos.

- **Desvantagens**: Os resultados podem ser difíceis de interpretar, e a validação do modelo é desafiadora devido à falta de rótulos.

#### 3. Aprendizado por Reforço
#### Definição
No aprendizado por reforço, um agente aprende a tomar decisões através de interações com um ambiente. O agente recebe recompensas ou penalidades com base nas ações que realiza, e o objetivo é maximizar a recompensa total ao longo do tempo. Chamamos de política as ações que o agente toma dado um estado observado.

#### Técnicas Comuns

- **Q-Learning**: Uma técnica que busca encontrar a política ótima de ações para maximizar a recompensa total.

- **Política de Gradiente**: Método que otimiza diretamente a política de tomada de decisões.

#### Vantagens e Desvantagens

- **Vantagens**: Muito eficaz em problemas onde a tomada de decisões sequenciais é crucial, como jogos ou robótica.

- **Desvantagens**: Requer muitas interações com o ambiente para treinar, pode ser computacionalmente caro e difícil de configurar.

Ainda existem outros tipos de aprendizado menos comuns a ser explorados, como o aprendizado Semi-Supervisionado ou Auto-Supervisionado, que serão em outros capítulos do livro por conta de sua complexidade.

Nos próximos capítulos exploraremos o nosso primeiro modelo de *Machine Learning*: Regressão Linear (Sim, do Aprendizado Supervisionado). Até lá!

## Recursos Úteis

- [Tipos de Aprendizado de Máquina | Peixe Babel 53](https://www.youtube.com/watch?v=YuUIxpCA-EQ)