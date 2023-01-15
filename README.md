# CVDL
Small change of an implementation of the CAAE taken from another repo.

This implementation changes:
  - for the training part - the input_output loss function of the model is changed from L1 to L2
  - for the output - the original output generated an image that projects the input on each age category
                   - i modified it so it only produces one image as the output for a given age group
                   
Original repo : https://github.com/mattans/AgeProgression
