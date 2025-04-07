options(scipen = 999)

#0=lascia (churn) ---> evento
#1= resta---> non evento

setwd("/Users/nicolomelchioretto/Desktop/UniMIB/Corsi/SGI/Anno_3/Data Mining e Machine Learning/Machine Learning/Tesina_ML")

#Importazione dataset

c <- read.csv("ricodifica.csv", sep = "," , dec = ".",  
              stringsAsFactors = TRUE, na.strings = c("NA","NaN", ""))

#Conversione variabili
c$Exited <- as.factor(c$Exited)
c$Complain <- as.factor(c$Complain)
c$Surname <- as.character(c$Surname)
c$IsActiveMember <- as.factor(c$IsActiveMember)
c$HasCrCard <- as.factor(c$HasCrCard)


#Rimuoviamo RowNumber, CustomerId, Surname
r <- c[,-c(1,2,3)]

#Descrizione dataset
summary(r)
str(r)
dim(r)
nrow(r)
names(r) #nomi delle variabili

library(Hmisc)
describe(r) 

library(funModeling)
library(dplyr)
status <- df_status(r, print_results = T) 
#Ordiniamo le variabili in status (raggruppate per type) per numero di missing e per numero di valori unici (ovviamente ha senso guardare le factor e integer)
head(status) %>% arrange(type, -p_na)
head(status) %>% arrange(type, -unique)
#Tutto a posto

#Descrizione target Exited nel dataset completo
table(r$Exited) 
(table(r$Exited))/nrow(r)*100 #percentuale di Exited #PRIOR

#Il dataset non è stato bilanciato in precedenza e l'evento di churn non è raro, quindi non dobbiamo fare niente a riguardo

#### PREPROCESSING ###

### 1. VALORI MANCANTI ###
sapply(r, function(x)(sum(is.na(x)))) #nessun missing


### 2. COLLINEARITÀ ###

#Estrazione factor
isfactor <- sapply(r, function(x) is.factor(x)) ; isfactor 
factordata <- r[, isfactor] ; head(factordata) #dataset con solo le factor
names(factordata)
sapply(factordata, function(x) length(levels(x))) #numero di livelli di ciascuna factor

#Estrazione numeriche
isnum <- sapply(r, function(x) is.numeric(x)) ; isnum 
numdata <- r[, isnum] ; head(numdata) 
names(numdata)

#Qualitative
library(plyr) 
library(dplyr)
combos <- combn(ncol(factordata),2)
adply(combos, 2, function(x) {
  test <- chisq.test(factordata[, x[1]], factordata[, x[2]])
  tab  <- table(factordata[, x[1]], factordata[, x[2]])
  out <- data.frame("Row" = colnames(factordata)[x[1]]
                    , "Column" = colnames(factordata[x[2]])
                    , "Chi.Square" = round(test$statistic,3)
                    , "df"= test$parameter
                    , "p.value" = round(test$p.value, 3)
                    , "n" = sum(table(factordata[,x[1]], factordata[,x[2]]))
                    , "u1" =length(unique(factordata[,x[1]]))-1
                    , "u2" =length(unique(factordata[,x[2]]))-1
                    , "nMinu1u2" =sum(table(factordata[,x[1]], factordata[,x[2]]))* min(length(unique(factordata[,x[1]]))-1 , length(unique(factordata[,x[2]]))-1) 
                    , "Chi.Square norm"  =test$statistic/(sum(table(factordata[,x[1]], factordata[,x[2]]))* min(length(unique(factordata[,x[1]]))-1 , length(unique(factordata[,x[2]]))-1)) 
  )
  
  
  return(out)
  
}) 

#Unico chi square alto è tra Exited e Complain = 0.99 : rimuoviamo Complain
s2 <- r[,-12]
#NB: Complain è correlata proprio con la variabile target! Quindi è il caso di una target mascherata da covariata (spiegherebbe tutto lei ma non utile ai fini di una analisi più approfondita delle cause di churn dei clienti)

#Quantitative con VIF
y <- as.numeric(s2$Exited) 
X <- numdata
X_matrix = as.matrix(X)
mod <- lm(y ~ X_matrix) #in notazione matriciale
library(mctest)
imcdiag(mod)

# tutte le var VIF sono vicine all’1, tutte minori di 5 
corrgram(numdata)
#Altrimenti, con caret:
R <- cor(numdata) ; R

library(caret)
correlatedPredictors = findCorrelation(R, cutoff = 0.8, names = TRUE) ; correlatedPredictors

#Summary della parte triangolare superiore della matrice di correlazione
summary(R[upper.tri(R)])

### 3. ZERO VARIANCE ###
library(caret)
nzv = nearZeroVar(s2, saveMetrics = TRUE)
nzv   
#nessuna variabile ha varianza vicino a zero 

### 4. SEPARATION ###

# controllo separation o quasi separation delle variabili indipendenti con y 
table(r$Exited, r$Card.Type) / nrow(r) # a posto
table(r$Exited, r$Gender) / nrow(r) # va bene
table(r$Exited, r$Geography) / nrow(r) # va bene


#ESTRAZIONE DATI DI SCORE
set.seed(123)
split_0 <- createDataPartition(y=s2$Exited, p = 0.9, list = FALSE)     
dati <- s2[split_0,]
score <- s2[-split_0,]

#Plot dei dati 
featurePlot(x = dati[, colnames(numdata)], 
            y = dati$Exited, 
            plot = "box", #utile anche "density" per mostrare le densità 
            scales = list(y = list(relation="free"),
                          x = list(rot = 0, relation = "free")),  
            auto.key = list(columns = 3))
?featurePlot

### DIVIDO IN TRAINING E TEST ###

library(caret)
set.seed(123)
split <- createDataPartition(y=dati$Exited, p = 0.70, list = FALSE)     
train <- dati[split,]
test <- dati[-split,]

###TUNING DEI MODELLI###

##### PRE-PROCESSING ####

#1) Missing data: serve per tutti (tranne tree, rf, gb) ma non ce ne sono quindi a posto 
#2) nzv : serve per tutti (tranne tree, rf, gb) ma non ce n'è sono quindi a posto
#3) collinearity: serve per logistico, nb, reti neurali: giù risolta con Complain quindi a posto
#4) model selection: rf, tree, gb, nb non la richiedono; logistico, knn, nnet, pls si => da fare 
#5) outlier: serve per logistico e reti MA bordello (serve il modello ecc) QUINDI se vediamo che i modelli fanno schifo potrebbe essre colpa degli outlier e torniamo indietro e lo facciamo (magari con preProcess di caret)
#6) std/norm variabili: lasso, reti neurali, pls , knn
#7) separation/zero-problem: logistico, nb => da fare 
?preProcess
#Soluzione: due dataset (di training): uno per tree,rf,gb; l'altro per gli altri 

#NB: DISCORSO DELLA CODIFICA DEL TARGET
#Per alcuni serve il target factor per altri numerico; QUINDI creiamo due verisioni di train/test: una con target numerico l'altra con factor

#Train/test con target factor

library(car)
train$Exited <- recode(train$Exited, recodes = "0 = 'c0'; else = 'c1' ") 
str(train$Exited) 
head(train$Exited)

test$Exited <- recode(test$Exited, recodes = "0 = 'c0'; else = 'c1' ") 
str(test$Exited) 
head(test$Exited)

train_f <- train
test_f <- test

#Train/test con targtet numerico

train$Exited <- ifelse(train$Exited == "c0", 0, 1)
str(train$Exited) 
head(train$Exited)

test$Exited <- ifelse(test$Exited == "c0", 0, 1)
str(test$Exited) 
head(test$Exited)

train_n <- train
test_n <- test

(table(train_n$Exited))/nrow(train_n)*100 #percentuale di Exited nel train #PRIOR

#### [1] TUNING MODELLI ####

#CARET LEGGE COME CLASSE 1 LA CLASSE C0 => TUNARE PER LA SENS = TUNARE PER C0 

#Caso peggiore: classificare gli 0 come 1 (cioè classificare come "non exit" gli "exit") perchè la banca non riconosce quali clienti perderà
#Quindi: tuniamo con la Sensitivity così da allenare il modello a classificare bene i true positive, cioè i clienti che lasceranno la banca 

##### SENZA PRE-PROC ####

#Tuning dei modelli che non richiedono pre-processing 

###### TREE #### 
set.seed(1234)
metric <- "Sens"
ctrl_tree <- trainControl(method = "repeatedcv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE) 
tree <- train(Exited ~ . , data = train_f, method = "rpart", 
              tuneLength = 10, trControl = ctrl_tree, minsplit = 1, metric = metric) 

tree #mostra il risulta per ciscun valore di cp

#Performance sul validation e verifica overfitting
library(caret)
tree_pred <- predict(tree, test_f, type = "raw")
confusionMatrix(tree_pred, as.factor(test_f$Exited))

#Variable importance dell'albero
var_imp_tree <- varImp(tree)
plot(var_imp_tree) 

#Plot
library(rpart)
library(rpart.plot)
tree_tuned <- rpart(Exited ~ . , data = train_f, method = "class", 
                    cp = 0.00311284, minsplit = 1, xval = 5)
rpart.plot(tree_tuned, type = 4, extra = 1)

#Altro albero prunato più sopra

set.seed(1234)
metric <- "Sens"
ctrl_tree_2 <- trainControl(method = "repeatedcv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE) 
tree2 <- train(Exited ~ . , data = train_f, method = "rpart", 
               tuneLength = 10, trControl = ctrl_tree_2, minsplit = 200, metric = metric) 

tree2 #mostra il risultato per ciscun valore di cp

#Performance sul validation e verifica overfitting (esattamente identico a quello prima non prunato)
confusionMatrix(tree2)
library(caret)
tree2_pred <- predict(tree2, test_f, type = "raw")
confusionMatrix(tree2_pred, as.factor(test_f$Exited))
confusionMatrix(tree2)


#Variable importance dell'albero
var_imp_tree2 <- varImp(tree2)
plot(var_imp_tree2) 

#Plot
library(rpart)
library(rpart.plot)
tree2_tuned <- rpart(Exited ~ . , data = train_f, method = "class", 
                     cp = 0.00311284, minsplit = 200, xval = 5)
rpart.plot(tree2_tuned, type = 4, extra = 1)

###### RANDOM FOREST ####

library(caret)
set.seed(1234)
metric <- "Sens"   
ctrl_rf <- trainControl(method = "repeatedcv", number = 10, search = "grid", summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE)
grid_rf <- expand.grid(.mtry = seq(1, 16, 2)) #grid per il parametro di tuning k 
rf <- train(Exited ~ . , data = train_f , method = "rf", metric = metric, tuneGrid = grid_rf, ntree = 250, trControl = ctrl_rf)
rf

#Performance su validation e overfitting

library(caret)
confusionMatrix(rf)
rf_pred <- predict(rf, test_f, type = "raw")
confusionMatrix(rf_pred, as.factor(test_f$Exited))

#Plot
plot(rf)

#Performance della rf su varie metriche per ciascun valore di k (mtry) 
rf$results

#Performance su metriche su ciascun fold del modello migliore
rf$resample

#Performance del modello finale come media sulle fold
mean(rf$resample$ROC)
mean(rf$resample$Spec)
mean(rf$resample$Sens)
ls(rf$resample)

#Variable importance nella RF
var_imp_rf <- varImp(rf) 
head(var_imp_rf)
plot(var_imp_rf) #SI PRWENDE UNA SOGLIA E SI USA RF COME MODEL SELECTOR: LE VAR CON VAR IMP OLTRE LA SOGLIA SI DANNO IN PASTO AD ALTRI ALGORITMI CHE RICHIEDONO MODEL SELECTION

#Salviamo la variable importance in un dataframe
var_imp_rf.df <- as.data.frame(var_imp_rf$importance)
head(var_imp_rf.df)

###### GRADIENT BOOSTING ####

set.seed(1234) 
metric <- "Sens"   
ctrl_gbm <- trainControl(method = "repeatedcv", number = 10, summaryFunction = twoClassSummary, search = "grid", classProbs = TRUE, savePredictions = TRUE)
gbm_grid <- expand.grid(n.trees = 200, interaction.depth = c(1:9), shrinkage = c(0.075, 0.1, 0.5, 0.7), n.minobsinnode = 20) #????valori messi a caso o da tunare fittando già lpalbero singolo come modello (soprattuto minobs e interaction.depth)??????
gbm <- train(Exited ~ . , data = train_f, method = "gbm", trControl = ctrl_gbm, tuneGrid = gbm_grid, metric = metric, verbose = FALSE) 
gbm

#Performance su validation e overfitting
library(caret)
confusionMatrix(gbm)
gbm_pred <- predict(gbm, test_f, type = "raw")
confusionMatrix(gbm_pred, as.factor(test_f$Exited))

#Plottiamo al variare dei vari parametri
ggplot(gbm)

###### BAGGING ####

library(caret)
set.seed(1234)
metric = "Sens"
ctrl_bag = trainControl(method = "repeatedcv", number = 10, search = "grid", summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE)
bag = train(Exited ~ . , data = train_f, method = "treebag", ntree = 250, 
            trControl = ctrl_bag)
bag

#Performance su validation e overfitting
confusionMatrix(bag)
bag_pred = predict(bag, test_f, type = "raw")
confusionMatrix(bag_pred, test_f$Exited)

##### CON PRE-PROC ####

####### BORUTA MODEL SELECTION #####

library(Boruta)
set.seed(123)
boruta <- Boruta(Exited ~ . , data = train_f, doTrace = 1) 

#Plottiamo i box-plot dell'importanza (MDI in realtà) delle variabili (sono box-plot a causa delle varie iterazioni, quindi l'MDI ha una distribuzione)
plot(boruta, xlab = "Features", xaxt = "n", ylab = "MDI")
#Rosso: rejected; Giallo: tentative; Verde: confirmed

#Risultato di Boruta
print(boruta)

#Metriche di Boruta per ogni variabile
boruta.metrics <- attStats(boruta)
boruta.metrics
table(boruta.metrics$decision) #Decisioni 

#Model selection: selezioniamo le variabili
var_selected <- subset(boruta.metrics, decision == "Confirmed")
var_selected
sel <- t(var_selected) ; sel

#Estraiamo ora le variabili selezionate nel dataset originale
boruta_selected_f = train_f[,colnames(sel)]
dim(boruta_selected_f)

boruta_selected_n = train_n[,colnames(sel)]
dim(boruta_selected_n)

#Riaggiaciamo la variabile target ai due train

train_sel_f <- cbind(train_f$Exited, boruta_selected_f); head(train_sel_f)
train_sel_n <- cbind(train_n$Exited, boruta_selected_n); head(train_sel_n)
colnames(train_sel_f)[1] <- "Exited" #Rinominiamo la colonna del target
colnames(train_sel_n)[1] <- "Exited"

test_sel_f <- test_f[,colnames(train_sel_f)]; head(test_sel_f)
test_sel_n <- test_n[,colnames(train_sel_n)]; head(test_sel_n)

#Salviamo i dataset con model selection in un file così da poterlo caricare senza fare ogni volta boruta
save(train_sel_f, test_sel_f, train_sel_n, test_sel_n, file = "train&test_f_n_borutamodsel.RData")
load("train&test_f_n_borutamodsel.RData")

###### LOGISTICO ####

library(caret)
set.seed(1234)
metric <- "Sens"
ctrl_glm <- trainControl(method = "repeatedcv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE)
glm <- train(Exited ~ . , data = train_sel_f , method = "glm", preProcess = c("bagImpute", "BoxCox"), metric = metric,
             trControl = ctrl_glm, tuneLength = 5, trace = FALSE)
glm

#Predictions e verifica overfitting
confusionMatrix(glm)
glm_pred <- predict(glm, test_sel_f, type = "raw")
confusionMatrix(glm_pred, as.factor(test_sel_f$Exited))

###### LASSO ####

library(glmnet)
library(caret)
set.seed(1234)
ctrl_lasso <- trainControl(method = "repeatedcv", number = 10, classProbs = T, savePredictions = TRUE, summaryFunction = twoClassSummary)
grid_lasso <- expand.grid(.alpha = 1, .lambda = seq(0, 1, by = 0.01)) 
metric <- "Sens"
lasso <- train(Exited ~ . , data = train_f, method = "glmnet", trControl = ctrl_lasso, tuneLength = 5, na.action = na.pass, preProcess = c("center", "scale"), metric = metric,
               tuneGrid = grid_lasso)
lasso

#Predictions e verifica overfitting
confusionMatrix(lasso)
lasso_pred <- predict(lasso, test_f, type = "raw")
confusionMatrix(lasso_pred, as.factor(test_f$Exited))

###### PLS ####

#Fittiamo una regressione tramite Partial Least Squares con caret
#NB: anche qui serve del preprocessing: imputation, model selection, variabili standardizzate/normalizzate

library(caret)
library(pls)
set.seed(1234)
ctrl_pls <- trainControl(method = "repeatedcv", number=10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
metric <- "Sens"
pls <- train(Exited ~ . , data = train_sel_f, method = "pls", metric = metric, preProcess = c("center", "scale"),
             trControl = ctrl_pls, tuneLength = 5)
pls

#Predictions e verifica overfitting
confusionMatrix(pls)
pls_pred <- predict(pls, test_sel_f, type = "raw")
confusionMatrix(pls_pred, as.factor(test_sel_f$Exited))

###### NAIVE BAYES ####

library(caret)
set.seed(1234)
ctrl_nb <- trainControl(method = "repeatedcv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)
metric <- "Sens"
nb <- train(Exited ~ . , data = train_f, method = "nb", metric = metric, 
            trControl = ctrl_nb, tuneLength = 5)
nb
plot(nb)
#Predictions e verifica overfitting
confusionMatrix(nb)
nb_pred <- predict(nb, test_f, type = "raw")
confusionMatrix(nb_pred, as.factor(test_f$Exited))


###### K-NN ####

set.seed(1234)
ctrl_knn <- trainControl(method = "repeatedcv", number = 10, classProbs = T, summaryFunction = twoClassSummary, savePredictions = TRUE)
grid_knn = expand.grid(k = seq(5,20,3))
metric = "Sens"
knn = train(Exited ~ . , data = train_sel_f, method = "knn",
            trControl = ctrl_knn, tuneLength = 5, metric = metric,
            tuneGrid = grid_knn, preProcess = c("scale", "center"))
knn
plot(knn)

#Predictions e verifica overfitting
confusionMatrix(knn)
knn_pred <- predict(knn, test_sel_f, type = "raw")
confusionMatrix(knn_pred, as.factor(test_sel_f$Exited))


###### NEURAL NETWORK ####

library(caret)
set.seed(1234)
metric <- "Sens" 
grid_nnet <- expand.grid(size=c(1:5), decay = c(0.001, 0.01, 0.05 , .1, .3))
ctrl_nnet = trainControl(method = "repeatedcv", number = 10, search = "grid", classProbs = T, summaryFunction = twoClassSummary, savePredictions = TRUE)
nnet <- train(Exited ~ . , data = train_sel_f, method = "nnet", tuneGrid = grid_nnet,
              preProcess = c("center", "scale"), metric = metric, trControl = ctrl_nnet,
              trace = TRUE, # use true to see convergence
              maxit = 300)
print(nnet)
plot(nnet) 
getTrainPerf(nnet) 

#Predictions e verifica overfitting
confusionMatrix(nnet)
nnet_pred <- predict(nnet, test_sel_f, type = "raw")
confusionMatrix(nnet_pred, as.factor(test_sel_f$Exited))

##### MODELLI ENSEMBLE ####

## COMPARISON STEP 1 ##
results <- resamples(list(tree=tree, tree2 = tree2, rf = rf, gbm = gbm, bag = bag, glm = glm, lasso = lasso, pls = pls, nb = nb, knn = knn, nnet = nnet))

#Riassumiamo le distribuzioni delle metriche sulle folds dei vari modelli
summary(results)

#Boxplots dei risultati
bwplot(results)
dotplot(results)
###nnet, rf, bagging, gbm sono i migliori sulla Spec; li useremo quindi nell'ensemble

#Matrice di differenze delle acc, sens, spec tra i vari modelli
diffs <- diff(results)
summary(diffs)


###### ENSEMBLE ####

library(caretEnsemble) 

#Creiamo una lista, convertita poi in formato careList per darla in pasto a caretEnsamble 

best <- list("rf" = rf, "gbm" = gbm, "nnet" = nnet, "bag" = bag, "knn"= knn) 
class(best) <- "caretList" ; class(best)

#Facciamo l'ensemble usando una metrica a scelta
set.seed(1234)
metric <- "Sens"
control_ens <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE)
ens <- caretEnsemble(best, metric = metric, trControl = control_ens)
ens
plot(ens)

#Ricaviamo le predictions sul validation 
ens_pred <- predict(ens, test_f, type = "raw")
confusionMatrix(ens_pred, as.factor(test_f$Exited))

#Ricaviamo le predictions sul validation in formato posterior (che useremo poi per fittare le ROC)
ens_pred_roc <- predict(ens, test_f, type = "prob") ; head(ens_pred_roc)


###### STACKING ####

library(caretEnsemble) 

#Usiamo la caretList ricavata nell'Ensemble
best

#Facciamo lo stacking usando una metrica a scelta e un logistico come meta-classifier specificandolo in method

set.seed(1234)
metric <- "Sens"
ctrl_stack <- trainControl(method = "repeatedcv", number = 10, summaryFunction = twoClassSummary, classProbs = TRUE, savePredictions = TRUE)
stack_glm <- caretStack(best, method = "glm", metric = metric, trControl = ctrl_stack)
stack_glm
#Summary della logistca meta-modello

summary(stack_glm)

#Ricaviamo le predictions sul validation 
stack_pred <- predict(stack_glm, test_f, type = "raw"); head(stack_pred)
confusionMatrix(stack_pred, as.factor(test_f$Exited))

#Ricaviamo le predictions sul validation in formato posterior (che useremo poi per fittare le ROC)
stack_pred_roc <- predict(stack_glm, test_f, type = "prob"); head(stack_pred_roc)


#### [2] MODEL ASSESSMENT ####


###### ROC ####

test_f_roc <- test_f
test_f_sel_roc <- test_sel_f

# poichè vogliamo le probabilità di c0 (il nostro evento) che sono nella colonna 1 nei modelli non "aggregati",
# in stacking e ensamble, siccome non abbiamo due colonne ma solo un vettore contenente la prob di c0, 
# per avere le prob di c1 calcoliamo (1 - prob di c1)

#per stacking
#head(stack_pred_roc)
#stack_pred_roc_2 <- 1 - stack_pred_roc
#head(stack_pred_roc_2)

#per ensamble 
#head(ens_pred_roc)
#ens_pred_roc_2 <- 1 - ens_pred_roc
#head(ens_pred_roc_2)

#Ricaviamo le posterior

#NB: mettiamo ,1 poichè ci interessa la c0 (i nostri churn) che stanno sulla colonna 1
test_f_roc$p1 <- predict(tree2       , test_f, "prob")[,1]
test_f_roc$p2 <- predict(rf         , test_f, "prob")[,1]
test_f_roc$p3 <- predict(bag    , test_f, "prob")[,1]
test_f_roc$p4 <- predict(gbm      , test_f, "prob")[,1]
test_f_roc$p5 <- stack_pred_roc
test_f_roc$p6 <- ens_pred_roc
test_f_roc$p7 <- predict(lasso      , test_f, "prob")[,1]
test_f_roc$p8 <- predict(nb      , test_f, "prob")[,1]
test_f_sel_roc$p9 <- predict(pls    , test_sel_f, "prob")[,1]
test_f_sel_roc$p10 <- predict(glm      , test_sel_f, "prob")[,1]
test_f_sel_roc$p11 <- predict(knn, test_sel_f, "prob")[,1]
test_f_sel_roc$p12 <- predict(nnet, test_sel_f, "prob")[,1]

library(pROC)
# roc values
r1 <- roc(Exited ~ p1, data = test_f_roc)
r2 <- roc(Exited ~ p2, data = test_f_roc)
r3 <- roc(Exited ~ p3, data = test_f_roc)
r4 <- roc(Exited ~ p4, data = test_f_roc)
r5 <- roc(Exited ~ p5, data = test_f_roc)
r6 <- roc(Exited ~ p6, data = test_f_roc)
r7 <- roc(Exited ~ p7, data = test_f_roc)
r8 <- roc(Exited ~ p8, data = test_f_roc)
r9 <- roc(Exited ~ p9, data = test_f_sel_roc)
r10 <- roc(Exited ~ p10, data = test_f_sel_roc)
r11 <- roc(Exited ~ p11, data = test_f_sel_roc)
r12 <- roc(Exited ~ p12, data = test_f_sel_roc)
??plot

#Plottiamo le ROC
plot(r1, main = "Model Assessment")
plot(r2, add = T, col = "red")
plot(r3, add = T, col = "blue")
plot(r4, add = T, col = "yellow")
plot(r5, add = T, col = "violet")
plot(r6, add = T, col = "green")
plot(r7, add = T, col = "orange")
plot(r8, add = T, col = "purple")
plot(r9, add = T, col = "darkgreen")
plot(r10, add = T, col = "gray")
plot(r11, add = T, col = "skyblue")
plot(r12, add = T, col = "brown")
?legend

legend(title = "ROC Curves", "right", 
       legend = c("tree", "rf", "bag", "gbm", "stack_glm", "ens", "lasso", "nb", "pls", "glm", "knn", "nnet" ), 
       bty = "n", cex = 0.7, lty = 1, 
       col = c("black", "red", "blue", "yellow", "violet", "green", "orange", "purple", "darkgreen", "gray","skyblue","brown"))

#NB: ROC di stacking e ensemble si sovrappongono praticamente alla perfezione...dato che sono due modelli ensemble si potrebbe levare Ensemble e tenere solo Stacking

###### LIFT #### 

#lift utile per confrontare i modelli quando le curve roc si intersecano
# scegli i modelli con le migliori prestazioni sui primi 2,3 decili 
#(sui migliori 20, 30% , 40% dei casi, quelli con maggiore probabilità di churn)

library(funModeling)
gain_lift(data = test_f_roc, score = 'p2', target = 'Exited') # lift della random forest
gain_lift(data = test_f_roc, score = 'p5', target = 'Exited') # lift dello stacking
gain_lift(data = test_f_roc, score = 'p6', target = 'Exited') # lift dell'ensamble
gain_lift(data = test_f_sel_roc, score = 'p12', target = 'Exited') # lift di nnet
gain_lift(data = test_f_sel_roc, score = 'p11', target = 'Exited')# lift di knn

#gurdando il terzo decile il moglior modello: sono sia l'ensable che lo stacking
#Eliminare uno tra: ensable e stacking (SONO IDENTICI)
#scegliamo lo staking perchè ha due livelli di modelli (coinvolge un metamodello)
#nel 3° decile (quindi con il 30% dei dati) lo stacking classifica correttamente come c0 (churner) il 75,4% dei casi


#### [3] SCELTA DELLA SOGLIA####

y <- test_f_roc$Exited
y <- ifelse(y == "c0", 1, 0) 
predProbST <- test_f_roc$p2

library(ROCR)
predRoc <- prediction(predProbST, y) #serve per avere oggetto di classe rocr
class(predRoc)

###### CALCOLO E PLOT METRICHE/SOGLIA ####

#Calcoliamo e plottiamo (singolarmente) le metriche al variare della soglia
??performance

#spec
spec.perf = performance(predRoc, measure = "spec", x.measure = "cutoff")
plot(spec.perf, col = "red")

#acc
acc.perf = performance(predRoc, measure = "acc")
plot(acc.perf, col = "blue")

#precision (ppv = Positive predictive value = TP/(TP+FP))
prec.perf = performance(predRoc, measure = "prec")
plot(prec.perf, col = "purple")

#npv (Negative predictive value = TN/(TN+FN) )
npv.perf = performance(predRoc, measure = "npv")
plot(npv.perf, col = "violet")

#sens
sens.perf = performance(predRoc, measure = "sens")
plot(sens.perf, col = "green")

#f1
f1.perf = performance(predRoc, measure = "f")
plot(f1.perf, col = "darkgreen")

#Plot di tutte le metriche al variare della soglia

plot(spec.perf, col = "red", main = "Metriche al variare della soglia", ylab = "Metriche")
plot(acc.perf, add = T, col = "blue")
plot(prec.perf, add = T, col = "purple")
plot(npv.perf, add = T, col = "violet")
plot(sens.perf, add = T, col = "green")
plot(f1.perf, add = T, col = "darkgreen")
??plot
legend(title = "Metriche", "bottom", 
       legend = c("spec", "acc", "prec", "npv", "sens", "f1"), 
       bty = "n", cex = 0.5, lty = 1, 
       col = c("red", "blue", "purple", "violet", "green", "darkgreen"))


## Plot alternativo, per cui servono perfomance e cutoff in dataframe ##

cut=as.data.frame(spec.perf@x.values)
colnames(cut)="cutoff"
head(cut)

spec=as.data.frame(spec.perf@y.values)
colnames(spec)="spec"
head(spec)

cut_spec=cbind(cut,spec) 
head(cut_spec)

performance(predRoc,measure="spec")

sens=as.data.frame(sens.perf@y.values)
colnames(sens)="sens"

acc=as.data.frame(acc.perf@y.values)
colnames(acc)="acc"

prec=as.data.frame(prec.perf@y.values)
colnames(prec)="prec"

npv=as.data.frame(npv.perf@y.values)
colnames(npv)="npv"

f1=as.data.frame(f1.perf@y.values)
colnames(f1)="f1"


all = cbind(cut_spec, sens, acc, prec, npv, f1)

head(all)
dim(all)
tail(all)

library(reshape2)
metrics <- melt(all, id.vars = "cutoff", 
                variable.name = "Metriche",
                value.name = "Measure")
head(metrics)
dim(metrics)

#Plot metriche vs soglia
ggplot(metrics, aes(x = cutoff, y = Measure, color = Metriche)) + 
  geom_line(size=1) + 
  ylab("") + xlab("Probability Cutoff") +
  theme(legend.position = "top")


#Scegliamo 0.1 
#Oss: ha senso,dato che la sens individua in questo caso l'evento più raro (churn), quindi la soglia deve essere bassa
#use decision rule for the best model#####
pred_y <- ifelse(test_f_roc$p2 > 0.1, "c0", "c1")
pred_y <- as.factor(pred_y)
head(pred_y)

confusionMatrix(pred_y, test_f_roc$Exited)


### [4] CLASSIFICAZIONE DATI DI SCORE  #####

score$prob = predict(rf, score, "prob")
head(score$prob)
prob_c0 <- score$prob[,1]
score$pred_y <- ifelse(prob_c0 > 0.1, "c0", "c1")
head(score)

score$Exited <- ifelse(score$Exited == 0, "c0", "c1")
head(score)

score$Exited <- as.factor(score$Exited)
score$pred_y <- as.factor(score$pred_y)

confusionMatrix(score$pred_y, score$Exited)


#### INTERPRETAZIONE DELLA RF ####

#Diamo un'interpretazione degli effetti delle variabili sulla classe prevista dal modello, in questo caso Random Forest
#Di fatto equivalente a Partial plots dei modelli lineari
#Un Partial dependence profile dice di più rispetto alla variable importance della RF, ovvero la direzione e l'andamento della connessione tra covariate e target

library(DALEX)

str(train_f)
plot(var_imp_rf)
??DALEX::plot
#Prediction su Exited sul train

pred_rf_prob = predict(rf, train_f, type = "prob")[,1]
head(pred_rf_prob)

#Partial dependence plots sulle numeriche

explainer  <- explain(rf, data = train_f) 

sv1  <- single_variable(explainer, variable = "Age",  type = "pdb", prob = T)
plot(sv1)

sv2  <- single_variable(explainer, variable = "NumOfProducts",  type = "pdb", prob = T)
plot(sv2)

sv3  <- single_variable(explainer, variable = "Balance",  type = "pdb", prob = T)
plot(sv3)

sv4  <- single_variable(explainer, variable = "CreditScore",  type = "pdb", prob = T)
plot(sv4)

sv5  <- single_variable(explainer, variable = "Tenure",  type = "pdb", prob = T)
plot(sv5)

sv8  <- single_variable(explainer, variable = "EstimatedSalary",  type = "pdb", prob = T)
plot(sv8)

sv9  <- single_variable(explainer, variable = "Satisfaction.Score",  type = "pdb", prob = T)
plot(sv9)

sv10  <- single_variable(explainer, variable = "Point.Earned",  type = "pdb", prob = T)
plot(sv10)

?single_variable

#Possiamo anche avere un'altra versione della variable importance della Random Forest

predict.fun <- function(model, x) predict(model, x, type = "prob")[,1]

#copy <- train_n[-11]
explainer_rf  <- explain(rf, data = test_n, y = test_n$Exited, predict.function = predict.fun)
vd_rf <- variable_importance(explainer_rf, type = "raw")
vd_rf
plot(vd_rf)


#Ora valutiamo l'effette delle variabili sulla prediction della RF su una nuova singola osservazione presa dal dataset di score

#new_observation <- score[700,-c(11,15:17)]
new_observation
#Dalla obs corrispondente prenderà i valori che si ritrovano poi nel plot in basso

#Prediction sull'osservazione 

pred_prob_rf_newobs = predict(rf, score, type = "prob")[,1] ; head(pred_prob_rf_newobs)

#Posterior di churn sulla nuova osservazione
pred_prob_rf_newobs[700]

#Spiegazione della prediction
library("breakDown")
explain_3 <- broken(rf, new_observation, data = train_f, predict.function = predict.fun)
explain_3

library(ggplot2)
plot(explain_3) + ggtitle("breakDown plot new case for RF caret model")






















