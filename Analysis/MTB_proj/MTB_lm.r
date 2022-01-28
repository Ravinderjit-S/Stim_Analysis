library(tidyverse)
library(rmatio)
library(feather)
library(lme4)
library(lmerTest)
library(car)



dat_loc <- '/media/ravinderjit/Data_Drive/Data/MTB_dataframe'

dat <- read_feather(dat_loc)

#make factors
dat$age = as.factor(dat$age)


m <- lm(CMR ~ consec_coh + thresh_coh  + spaced_coh + CMRlapse + memr, data=dat)
summary(m)

ggplot(dat, aes(x = consec_coh + thresh_coh + spaced_coh + CMRlapse + memr , y = CMR)) + 
  geom_point() +
  stat_smooth(method = "lm", col = "red")


plot(Predict(m),data=dat)


m <- lm(MRT ~  thresh_coh + age, data=dat)
summary(m)

ggplot(dat, aes(x = thresh_coh + consec_coh + spaced_coh , y = Jane)) + 
  geom_point() +
  stat_smooth(method = "lm", col = "red")


pred <- predict.lm(m,se.fit=TRUE)
