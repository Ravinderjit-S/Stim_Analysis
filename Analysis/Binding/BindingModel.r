library(tidyverse)
library(rmatio)

vars_floc <- "/media/ravinderjit/Data_Drive/Data/EEGdata/MTB/Model/Binding20TonesModelVars.mat"

vars <- read.mat(vars_floc)
#spacedCoh, consecCoh, features
bind_data <- data.frame(vars)

colnames(bind_data) <- c('spacedCoh','consecCoh', 'Onset', 'AB12', 'BA12', 'AB20', 'BA20',
                                 'sus12AB', 'sus12BA', 'sus20AB', 'sus20BA')

ggplot(data=bind_data) +
  geom_point(mapping = aes(x = consecCoh, y=spacedCoh))

ggplot(data=bind_data) +
  geom_point(mapping = aes(x = BA20, y=spacedCoh))


plot(density(bind_data$spacedCoh))

boxplot(bind_data$sus20BA)

#drops <- c("consecCoh")
#consec_data = bind_data[ , !names(bind_data) %in% drops]

bind_data2 <- bind_data[-c(2),]
consec_model <- lm(spacedCoh ~  sus20AB + BA12 , data=bind_data)
summary(consec_model)


plot(bind_data$sus20AB,bind_data$spacedCoh)
abline(consec_model)


