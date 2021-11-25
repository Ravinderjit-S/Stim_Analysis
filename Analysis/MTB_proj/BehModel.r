library(tidyverse)
library(rmatio)

vars_floc <- "/media/ravinderjit/Data_Drive/Data/MTB_Behavior/CMR_Bind.mat"

vars <- read.mat(vars_floc)
beh_data <- data.frame(vars)

Subjects <- beh_data['Subjects']
#drops <- c("Subjects")
#beh_data <- beh_data[,!(names(beh_data) %in% drops)]

boxplot(beh_data$consec_coh)
boxplot(beh_data$CMR)

ggplot(data=beh_data) +
  geom_point(mapping = aes(x = consec_coh, y=spacedCoh))

ggplot(data=beh_data) +
  geom_point(mapping = aes(x = spacedCoh, y=CMR))

ggplot(data=beh_data) +
  geom_point(mapping = aes(x = consec_coh, y=CMR))

plot(density(beh_data$spacedCoh))
plot(density(beh_data$CMR))


beh_data2 <- beh_data[-c(21,22),]

cmr_model = lm(CMR ~  spacedCoh , data=beh_data2)
summary(cmr_model)

plot(beh_data2$spacedCoh,beh_data2$CMR)
abline(cmr_model)


bind_model = lm(spacedCoh ~  consec_coh , data=beh_data)
summary(bind_model)

plot(beh_data$consec_coh,beh_data$spacedCoh)
abline(bind_model)
