####################################
## Simple chain example: x -> y -> z
####################################
x = createCPT(list("x"), probs = c(0.3, 0.7), levelsList = list(c("T", "F")))
y.x = createCPT(list("y", "x"), probs = c(0.8, 0.4, 0.2, 0.6),
                levelsList = list(c("T", "F"), c("T", "F")))
z.y = createCPT(list("z", "y"), probs = c(0.5, 0.6, 0.5, 0.4),
                levelsList = list(c("T", "F"), c("T", "F")))

(xyzNet = list("x" = x, "y" = y.x, "z" = z.y))

## Some simple operations you might try to check your code
productFactor(x, y.x)
productFactor(productFactor(x, y.x), z.y)
marginalizeFactor(productFactor(x, y.x), "x")
marginalizeFactor(productFactor(y.x, z.y), "z")

## Notice in the observe function, you just need to delete rows that are
## inconsistent with the given observations. Factors do not need to be combined
## or normalized in this step.
observe(xyzNet, "x", "T")
observe(xyzNet, c("x", "y"), c("T", "T"))

## Marginalize must first combine all factors involving the variable to
## marginalize. Again, this operation may lead to factors that aren't
## probabilities.
marginalize(xyzNet, "x")
marginalize(xyzNet, "y")
marginalize(xyzNet, "z")
marginalize(xyzNet, c("x", "z"))

#############################
## Bishop book (Ch 8) example
#############################
b = createCPT(list("battery"), probs = c(0.9, 0.1), levelsList = list(c(1, 0)))
f = createCPT(list("fuel"), probs = c(0.9, 0.1), levelsList = list(c(1, 0)))
g.bf = createCPT(list("gauge", "battery", "fuel"),
                 probs = c(0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9),
                 levelsList = list(c(1, 0), c(1, 0), c(1, 0)))

carNet = list("battery" = b, "fuel" = f, "gauge" = g.bf)

## Some examples:
## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
productFactor(productFactor(b, f), g.bf)
productFactor(productFactor(g.bf, f), b)

marginalizeFactor(productFactor(g.bf, b), "gauge")
productFactor(marginalizeFactor(g.bf, "gauge"), b)

productFactor(marginalizeFactor(productFactor(g.bf, b), "battery"), f)
marginalizeFactor(productFactor(productFactor(g.bf, f), b), "battery")

marginalizeFactor(productFactor(marginalizeFactor(productFactor(g.bf, b), "battery"), f), "gauge")
marginalizeFactor(productFactor(marginalizeFactor(productFactor(g.bf, b), "battery"), f), "fuel")

## Examples computed in book (see pg. 377)
infer(carNet, c("battery", "fuel"), NULL, NULL)     ## (8.30)
infer(carNet, c("battery"), "fuel", 0)              ## (8.31)
infer(carNet, c("battery"), "gauge", 0)             ## (8.32)
infer(carNet, NULL, c("gauge", "battery"), c(0, 0)) ## (8.33)


###########################################################################
## Kevin Murphy's Example: http://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html
###########################################################################
c = createCPT(list("cloudy"), probs = c(0.5, 0.5),
              levelsList = list(c("F", "T")))
r.c = createCPT(list("rain", "cloudy"), probs = c(0.8, 0.2, 0.2, 0.8),
                levelsList = list(c("F", "T"), c("F", "T")))
s.c = createCPT(c("sprinkler", "cloudy"), probs = c(0.5, 0.9, 0.5, 0.1),
                levelsList = list(c("F", "T"), c("F", "T")))
w.sr = createCPT(list("wet", "sprinkler", "rain"),
                 probs = c(1, 0.1, 0.1, 0.01, 0, 0.9, 0.9, 0.99),
                 levelsList = list(c("F", "T"), c("F", "T"), c("F", "T")))

grassNet = list("cloudy" = c, "rain" = r.c, "sprinkler" = s.c, "wet" = w.sr)

## Test your infer() method by replicating the computations on the website!!
infer(grassNet, c('cloudy', 'rain'), c('wet'), 'T')
infer(grassNet, c('cloudy', 'sprinkler'), c('wet'), 'T')
infer(grassNet, c('cloudy', 'rain', 'sprinkler'), NULL, NULL)

###########################################################################
## Class example
###########################################################################
traffic = createCPT(list("traffic"), probs = c(0.5, 0.5), levelsList = list(c("F", "T")))
alarm = createCPT(list("alarm"), probs = c(0.5, 0.5), levelsList = list(c("F", "T")))
late = createCPT(list("late", "alarm", "traffic"),
                 probs = c(0.8, 0.2, 0.8, 0.6, 0.2, 0.8, 0.2, 0.4),
                 levelsList = list(c("F", "T"), c("F", "T"), c("F", "T")))

lateNet = list(traffic, alarm, late)
