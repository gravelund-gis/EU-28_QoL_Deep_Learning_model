# EU-28 QoL Deep Learning model
Keras/TensorFlow Deep Learning regression model of EU-28 Quality of Life (QoL) levels for year 2014,
using 11 socioeconomic indicators, retrieved from Eurostat, as predictors and 1 climatic comfort predictor.
The target is 2013 EUSILC QoL survey data (ground truth).

A model to quantitatively estimate the spatial well-being (QoL) distribution across the EU-28 member states down
to the municipal level. The model is weight-driven and based on Eurostat statistics on objective key QoL indicators,
identified by the 2016 Eurostat Analytical report on subjective well-being and the 2013 EU-SILC ad-hoc module on
well-being. Additionally, some Europe 2020 strategy targets of the European Commission, deemed to be important to a
sense of personal well-being, are included, such as the risk of poverty or social exclusion and advanced educational
attainment.

A climatic comfort component based on 1961-1990 climatic normals is added to estimate the importance of (a static)
climate to QoL. Thermal comfort levels are obtained using the Universal Thermal Climate Index (UTCI) and
Predicted Mean Vote (PMV), and overall climatic comfort levels are obtained as a weighted linear combination
based on the classical Tourism Climatic Index (TCI).

To evaluate the performance of the Deep Learning model, it is compared to the results from the 2013 EU subjective
well-being survey (the ground truth).

This code is also made available in a Jupyter notebook format.

For a more complete background, see: https://lup.lub.lu.se/student-papers/search/publication/8905790
