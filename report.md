
# Report
    
## English
    
### Vectorizers
    
- Bag of words, duration of fitting: 69.3617160320282 seconds
- Word to vector, duration of fitting: 7.121267795562744 seconds
- Fast Text, duration of fitting: 20.88947820663452 seconds

    
### Classifiers
    

#### Naive Bayes by bag_of_words

Duration of fitting: 0.004841804504394531 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   268    |    1     |
|      Negative      |    8     |   862    |
+--------------------+----------+----------+
```
                       
Accuracy of Naive Bayes-> 99.20983318700614%

#### Random forest classifier by bag_of_words

Duration of fitting: 2.8163928985595703 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   240    |    29    |
|      Negative      |    0     |   870    |
+--------------------+----------+----------+
```
                       
Accuracy of Random forest classifier-> 97.45390693590869%

#### Support Vector Classification by bag_of_words

Duration of fitting: 4.3186633586883545 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   231    |    38    |
|      Negative      |    5     |   865    |
+--------------------+----------+----------+
```
                       
Accuracy of Support Vector Classification-> 96.22475856014047%

#### Naive Bayes by fast_text

Duration of fitting: 0.005217552185058594 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |    29    |   240    |
|      Negative      |    2     |   868    |
+--------------------+----------+----------+
```
                       
Accuracy of Naive Bayes-> 78.75329236172081%

#### Random forest classifier by fast_text

Duration of fitting: 2.7391839027404785 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   221    |    48    |
|      Negative      |    12    |   858    |
+--------------------+----------+----------+
```
                       
Accuracy of Random forest classifier-> 94.73222124670764%

#### Support Vector Classification by fast_text

Duration of fitting: 0.33856654167175293 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   225    |    44    |
|      Negative      |    17    |   853    |
+--------------------+----------+----------+
```
                       
Accuracy of Support Vector Classification-> 94.64442493415277%

#### Naive Bayes by word2vec

Duration of fitting: 0.004810333251953125 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |    15    |   254    |
|      Negative      |    0     |   870    |
+--------------------+----------+----------+
```
                       
Accuracy of Naive Bayes-> 77.69973661106233%

#### Random forest classifier by word2vec

Duration of fitting: 2.471653461456299 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   241    |    28    |
|      Negative      |    5     |   865    |
+--------------------+----------+----------+
```
                       
Accuracy of Random forest classifier-> 97.1027216856892%

#### Support Vector Classification by word2vec

Duration of fitting: 0.2795419692993164 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   252    |    17    |
|      Negative      |    18    |   852    |
+--------------------+----------+----------+
```
                       
Accuracy of Support Vector Classification-> 96.92712906057946%

    
## Russian
    
### Vectorizers
    
- Bag of words, duration of fitting: 58.519213914871216 seconds
- Word to vector, duration of fitting: 7.851081132888794 seconds
- Fast Text, duration of fitting: 27.605867862701416 seconds

    
### Classifiers
    

#### Naive Bayes by bag_of_words

Duration of fitting: 0.007219552993774414 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   267    |    2     |
|      Negative      |    10    |   828    |
+--------------------+----------+----------+
```
                       
Accuracy of Naive Bayes-> 98.91598915989161%

#### Random forest classifier by bag_of_words

Duration of fitting: 4.322728633880615 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   226    |    43    |
|      Negative      |    2     |   836    |
+--------------------+----------+----------+
```
                       
Accuracy of Random forest classifier-> 95.9349593495935%

#### Support Vector Classification by bag_of_words

Duration of fitting: 6.416926383972168 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   221    |    48    |
|      Negative      |    3     |   835    |
+--------------------+----------+----------+
```
                       
Accuracy of Support Vector Classification-> 95.39295392953929%

#### Naive Bayes by fast_text

Duration of fitting: 0.010374307632446289 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |    17    |   252    |
|      Negative      |    0     |   838    |
+--------------------+----------+----------+
```
                       
Accuracy of Naive Bayes-> 77.23577235772358%

#### Random forest classifier by fast_text

Duration of fitting: 2.4390294551849365 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   209    |    60    |
|      Negative      |    23    |   815    |
+--------------------+----------+----------+
```
                       
Accuracy of Random forest classifier-> 92.5022583559169%

#### Support Vector Classification by fast_text

Duration of fitting: 0.35440516471862793 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   218    |    51    |
|      Negative      |    30    |   808    |
+--------------------+----------+----------+
```
                       
Accuracy of Support Vector Classification-> 92.6829268292683%

#### Naive Bayes by word2vec

Duration of fitting: 0.0044558048248291016 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |    0     |   269    |
|      Negative      |    0     |   838    |
+--------------------+----------+----------+
```
                       
Accuracy of Naive Bayes-> 75.70009033423668%

#### Random forest classifier by word2vec

Duration of fitting: 2.4404890537261963 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   223    |    46    |
|      Negative      |    12    |   826    |
+--------------------+----------+----------+
```
                       
Accuracy of Random forest classifier-> 94.76061427280939%

#### Support Vector Classification by word2vec

Duration of fitting: 0.2954561710357666 seconds

```                       
+--------------------+----------+----------+
| Predicted \ Actual | Positive | Negative |
+--------------------+----------+----------+
|      Positive      |   234    |    35    |
|      Negative      |    22    |   816    |
+--------------------+----------+----------+
```
                       
Accuracy of Support Vector Classification-> 94.85094850948511%

    
