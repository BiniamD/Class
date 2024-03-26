library(datasets)
data(cars)
head(cars)

plot(cars, col='blue', pch=20, cex=2, main="Relationship between Speed and Stopping Distance for 50 Cars",
	xlab="Speed in mph", ylab="Stopping Distance in feet")


set.seed(122)
speed.c = scale(cars$speed, center=TRUE, scale=FALSE)
mod1 = lm(formula = dist ~ speed.c, data = cars)
summary(mod1)

plot(fitted(mod1), residuals(mod1))
abline(h = 0, col = "red")
