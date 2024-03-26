# Load necessary package
library(stats)
library(readxl)
install.packages("reshape2")

library(reshape2)

# Load data
Two_Factor <- read_excel(file.choose(), sheet = "Two_Factor")

# View data
View(Two_Factor)


# Stack data
stacked <- melt(Two_Factor)

# View stacked data
View(stacked)

# Rename columns
colnames(stacked) <- c("Education", "Field", "Income")

# View stacked data
View(stacked)


# Perform ANOVA test
twoway <- aov(Income ~ Education + Field, data = stacked)

# View ANOVA table
summary(twoway)


interact <- aov(Income ~ Education * Field, data = stacked)

summary(interact)

anova(interact)

Interaction <- read_excel(file.choose(), sheet = "Interaction")

View(Interaction)


Interaction[,1] <- c(rep("No High School", 3), rep("High School", 3), rep("Bachelor’s", 3), rep("Master's", 3))

View(Interaction)

stacked <- melt(Interaction)

View(stacked)

colnames(stacked) <- c("Education", "Field", "Income")

View(stacked)

twoway <- aov(Income ~ Education + Field, data = stacked)

summary(twoway)

anova(twoway)
#########################################################33333

Debt_Payments  <- read_excel(file.choose(), sheet = "Debt_Payments")

View(Debt_Payments)

cor.test(Debt_Payments$Income, Debt_Payments$Debt, alternative = "two.sided")

#Since the p-value is approximately equal to zero, 
#we reject H0. At the 5% significance level, we conclude that the population correlation coefficient between Debt and Income differs from zero.

options(scipen = 999)

multiple <- lm(Debt ~ Income + Unemployment, data = Debt_Payments)

summary(multiple)

predict(multiple, data.frame(Income = 80, Unemployment = 7.5))



################################
# Load data
Professor <- read_excel("c:/users/17034/downloads/jaggia_BS_4e_ch17_Data_Files.xlsx", sheet = "Professor")

View(Professor)

library(fastDummies)

Professor <- dummy_cols(Professor, select_columns = c("Age", "Sex"))
# Create dummy variables


Professor$Age <- ifelse(Professor$Age == "Older", 1, 0)
Professor$Sex <- ifelse(Professor$Sex == "Male", 1, 0)

View(Professor)

 