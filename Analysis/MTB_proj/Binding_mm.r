library(tidyverse)
library(rmatio)
library(lme4)
library(car)

data_Inter <- '/media/ravinderjit/Data_Drive/Data/MTB_proj/BindInter_phys_mm.mat'
data_consec <- '/media/ravinderjit/Data_Drive/Data/MTB_proj/BindConsec_phys_mm.mat'


vars_Inter <- read.mat(data_Inter)
vars_consec <- read.mat(data_consec)

Inter_data <- data.frame(vars_Inter)
consec_data <- data.frame(vars_consec)

#make Factors
Inter_data$sid = as.factor(Inter_data$sid)
Inter_data$age = as.factor(Inter_data$age)
Inter_data$cond = as.factor(Inter_data$cond)

consec_data$sid = as.factor(consec_data$sid)
consec_data$age = as.factor(consec_data$age)
consec_data$cond = as.factor(consec_data$cond)

str(Inter_data)
str(consec_data)


m <- lmer(acc_con ~ cond + B_on1 + (1|sid),data=consec_data)
summary(m)
Anova(m, test.statistic = 'F')

#plot(x=predict(m,consec_data),consec_data$acc_con)
#abline(a=0, b=1)

m2 <- lmer(acc_inter ~ cond + B_mn1 +B_on1 + (1|sid),data=Inter_data)
summary(m2)
Anova(m2, test.statistic = 'F')

I


