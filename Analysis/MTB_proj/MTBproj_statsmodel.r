library(tidyverse)
library(rmatio)
library(feather)
library(lme4)
library(lmerTest)
library(car)

#data_floc <- "/media/ravinderjit/Data_Drive/Data/MTB_dataframe"
data_cmr <- '/media/ravinderjit/Data_Drive/Data/MTB_proj/CMR_mm.mat'
data_behbind <- '/media/ravinderjit/Data_Drive/Data/MTB_proj/BindBeh_mm.mat'

#data_mtb <- read_feather(data_floc)

vars_cmr <- read.mat(data_cmr)
vars_BindBeh <- read.mat(data_behbind)

cmr_data <- data.frame(vars_cmr)
BehMtb_data <- data.frame(vars_BindBeh)

#make factors
cmr_data[,1] = as.factor(cmr_data[,1])
#cmr_data[,2] = as.factor(cmr_data[,2])
cmr_data[,5] = as.factor(cmr_data[,5])
#cmr_data$SNR = as.ordered(cmr_data$SNR)

str(cmr_data)

BehMtb_data$sid = as.factor(BehMtb_data$sid)
BehMtb_data$dist = as.ordered(BehMtb_data$dist)
BehMtb_data$ncoh = as.ordered(BehMtb_data$ncoh)

str(BehMtb_data)

p <- ggplot(aes(x=SNR,y=acc,color=coh),data=cmr_data) + geom_boxplot()
p + xlab('SNR') + ylab('Accuracy')

p <- ggplot(aes(x=ncoh,y=acc,color=dist),data=BehMtb_data) + geom_boxplot()
p + xlab('Dist') + ylab('Accuracy')


m <- lmer(acc ~ SNR + coh + age + (1|sid), data=cmr_data)
summary(m)
Anova(m,test.statistic='F')
anova(m)
aa <- predict(m)

m_mtb <- lmer(acc ~ ncoh + dist + age + (1|sid),data=BehMtb_data)
summary(m_mtb)
Anova(m_mtb,test.statistic = 'F')
anova(m)

plot(m)

ggplot(aes(y=predict(m)),data=cmr_data

#model <- lm(CMR ~ spaced_coh + age + memr ,data=data_mtb)
#summary(model)
#plot(data_mtb$CMR,data_mtb$Jane)
#abline(model)

