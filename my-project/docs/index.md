# dipspeaks


**X-ray binaries are amazing!** In these extreme environments, a compact object—either a neutron star or black hole—draws in matter from a companion star, producing intense X-ray emissions. These systems offer a unique window into extreme physics, from the effects of strong gravity and relativistic jets to the presence of intense magnetic fields.

Light curves tells us how the radiation that a source emmits varyes with time. Some subtle features, as dips and peaks within the light curves can provide insights into the wind structure and/or into the accretion processes.

To aid in the study of these events, we introduce **dipspeaks**.

If you have any questions or need assistance, please feel free to reach out: graciela.sanjurjo@ua.es.



## Getting started

### Installation
---------------------------------------------------------


You can install the package directly from PyPI using pip: **pip install dipspeaks**.

Or download the code from [here](https://github.com/xragua/dipspeaks/releases/tag/0.2.9).

Some examples of their usage are presented [here](https://github.com/xragua/dipspeaks/tree/main/examples).

And here [here](https://github.com/xragua/dipspeaks) is the page of this package 


### Which dips and peaks are we talking about?
---------------------------------------------------------
If the radiation from a Neutron star trhespases clumps, high energy tend to continue its path unaltered, while low energy is abserved. A dip with higher depth in the low energy range than in the high energy range can be a clump signature.

Peaks were related to Raighleigh-Taylor inestabilities close to the magnetosphere or Bondi irregularities. 


The primary challenge in this type of analysis is to distinguish the noise from real artifacts. In this algorithm we propose an autoencoder-approach to flag high-provability dips and peaks within a light curve.

So, dive in our **EXAMPLES** where we show some interesting uses cases.

