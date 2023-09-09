# CNN-CD

## Overview

Most deep learning models focused on forecasting ENSO indices while the zonal distribution of sea surface temperature anomalies (SSTA) over the equatorial Pacific was overlooked. To provide accurate predictions for the SSTA zonal pattern, we developed a model through leveraging the merits of the cosine distance in constructing the convolutional neural network. **This model (CNN-CD) can skillfully predict the SSTA zonal pattern over the equatorial Pacific 1 year in advance**, remarkably outperforming current dynamical models.

Moreover, we found that **the sources for ENSO predictability at different lead times are distinct**. For the 10-month-lead predictions, the precursors in the north Pacific, south Pacific and tropical Atlantic play critical roles in determining the model behaviors; while for the 16-month-lead predictions, the initial signals in the tropical Pacific associated with the discharge-recharge cycle are essential. 

## Data

* [Train/Trans/Valid]

      Data used in the training/transfer learning/validtion stages.
* [HeatValue]
      
    Heat value in the actiavtion maps.
* [ENSO Precursor]

    Normalized index of three ENSO precursors in JFM.
* [Assess]

    Data used in the assessments and analyses on the CNN-CD model.

## model

* [Build_CNN-CD.py]

    The script builds and trains the CNN-CD model.
* [Transfer_CNN-CD.py]

    The script performs the transfer learning.
* [Obtain_ActivationMap.py]

    The script used to obatin the heatvalues in the activation map.

## Plot

Drawing scripts
