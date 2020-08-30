#################################################################
############### OPERA PACKAGE TUTORIAL ##########################
#################################################################
### @AUTHOR: DEEPESH SINGH
### @DATE: 9TH NOV 2017
### $PURPOSE: OPERA PACKAGE INTRODUCTION
#################################################################

# INSTALLING OPERA PACKAGE
if(!require(opera)){install.packages(opera)};library(opera)
if(!require(caret)){install.packages(caret)};library(caret)
if(!require(forecast)){install.packages(forecast)};library(forecast)

# LOAD TEST DATA
data(electric_load)
idx_data_test <- 620:nrow(electric_load)
data_train <- electric_load[-idx_data_test, ] 
data_test <- electric_load[idx_data_test, ] 

# PLOTTING THE DATA
attach(electric_load)
plot(Load, type = "l", main = "The electric Load")

plot(Temp, Load, pch = 16, cex = 0.5, main = "Temperature vs Load")

# FINDING: WHERE EVER TEMPERATURE GOES UP LOAD GOES UP AND VICE VERSA

plot(NumWeek, Load, pch = 16, cex = 0.5, main = "Annual seasonality")

# Week numbers 26-32 have lowest load means winter

#################################### BUILDING EXPERTS ################

### FIRST EXPERT: GAM (GENERALIZED ADDITIVE MODEL)

library(mgcv)
gam.fit <- gam(Load ~ s(IPI) + s(Temp) + s(Time) + 
                 s(Load1) + as.factor(NumWeek), data = data_train)
gam.forecast <- predict(gam.fit, newdata = data_test)


### SECOND EXPERT: GRADIENT BOOSTING


gbm.fit <- train(Load ~ IPI + IPI_CVS + Temp + Temp1 + Time + Load1 + NumWeek, 
                 data = data_train, method = "gbm")
gbm.forecast <- predict(gbm.fit, newdata = data_test)


### THIRD EXPERT: Auto regressive short term correction

# medium term model
medium.fit <- gam(Load ~ s(Time,k=3) + s(NumWeek) + s(Temp) + s(IPI), data = data_train)
electric_load$Medium <- c(predict(medium.fit), predict(medium.fit, newdata = data_test))
electric_load$Residuals <- electric_load$Load - electric_load$Medium

# autoregressive correction
ar.forecast <- numeric(length(idx_data_test))
for (i in seq(idx_data_test)) {
  ar.fit <- ar(electric_load$Residuals[1:(idx_data_test[i] - 1)])
  ar.forecast[i] <- as.numeric(predict(ar.fit)$pred) + electric_load$Medium[idx_data_test[i]]
}

#ar.forecast <- data.frame(ar.forecast)


# COMPARING 3 EXPERTS VS ORIGINAL

Y <- data_test$Load
X <- cbind(gam.forecast, ar.forecast, gbm.forecast)
matplot(cbind(Y, X), type = "l", col = 1:6, ylab = "Weekly load",
        xlab = "Week", main = "Expert forecasts and observations")


############################## CHECK EXPERTS PERFORMANCE #########################################

# Is a single expert the best over time ? Are there breaks ?
oracle.convex <- oracle(Y = Y, experts = X, loss.type = "square", model = "convex")

#oracle.convex <- oracle(Y = Y, experts = X, loss.type = "square", model = "expert")
plot(oracle.convex)
print(oracle.convex)


################## AGGREGATION AND BUILDING MODEL #############################

# Building model with polynomial potential aggregation
MLpol0 <- mixture(model = "MLpol", loss.type = "square")

MLpol <- MLpol0
for (i in 1:length(Y)) {
  MLpol <- predict(MLpol, newexperts = X[i, ], newY = Y[i])
}

summary(MLpol)
weights <- predict(MLpol, X, Y, type='weights')

plot(MLpol, pause = TRUE)

# Generating predictation 

MLpol <- predict(MLpol0, newexpert = X, newY = Y, online = TRUE)

################## RETRAINING AND FORECASTING #############################

dt <- read.csv("ts_visitors.csv")

dt <- ts(dt$United.Kingdom, start = c(1998,4), frequency = 4)

train <- window(dt, end = c(2009,4), start = c(1999,1))
test <- window(dt,start = c(2010,1))

forecast.period <- length(test)

################# Example 2: COMPLETE MODEL - TRAINING WITH 3 EXPERTS ######################

expert.1 <- forecast(auto.arima(train), h = forecast.period)
expert.2 <- forecast(ets(train), h = forecast.period)
expert.3 <- forecast(tbats(train), forecast.period)
#expert.3 <- stlf(train, h = forecast.period)

train.experts <- cbind(ARIMA = fitted(expert.1), ETS = fitted(expert.2), TBATS = fitted(expert.3))

test.experts <- cbind(ARIMA = expert.1$mean, ETS = expert.2$mean, TBATS = expert.3$mean)


#### RETRAINING MODEL
MLpol <- mixture(Y = train, experts = train.experts,
                 loss.type = "square", model = "MLpol")

oracle.convex <- oracle(Y = train, experts = train.experts,
                        loss.type = "square", model = "convex")

plot(oracle.convex)
print(oracle.convex)

# Checking the weights
weights <- predict(MLpol, train.experts, train, type = "weights")
head(weights)
tail(weights)

### Forecasting 

z <- ts(predict(MLpol, test.experts, y = null,
                online = F, type = "response"), start = c(2010,1),
        frequency = 4)

qc.data <- data.frame(cbind(z, test, ARIMA = expert.1$mean, ETS = expert.2$mean, TBATS = expert.3$mean))

MAPE_Opera <- abs(sum(qc.data$z) - sum(qc.data$test))/sum(qc.data$test)*100
MAPE_Opera

MAPE_TBATS <- abs(sum(qc.data$TBATS) - sum(qc.data$test))/sum(qc.data$test)*100
MAPE_TBATS

MAPE_ARIMA <- abs(sum(qc.data$ARIMA) - sum(qc.data$test))/sum(qc.data$test)*100
MAPE_ARIMA

MAPE_ETS <- abs(sum(qc.data$ETS) - sum(qc.data$test))/sum(qc.data$test)*100
MAPE_ETS

#qc <- data.frame(MAPE_Opera, MAPE_ETS, MAPE_ARIMA, MAPE_TBATS)
t <- data.frame(MAPE_Opera, MAPE_ETS, MAPE_ARIMA, MAPE_TBATS)
qc <- rbind(qc, t)
qc

################################################# END OF THE CODE ####################
install.packages("Metrics")
library(Metrics)

a <- as.vector(qc.data['z'])
b <- as.vector(qc.data['test'])

accuracy(qc.data$z, qc.data$test)


accuracy(a,b)

qc.data$time <- c(1:length(qc.data$z))

df <- melt(qc.data ,  id.vars = 'time', variable.name = 'series')

library(ggplot2)
ggplot(df, aes(time,value)) + geom_line(aes(colour = series))
