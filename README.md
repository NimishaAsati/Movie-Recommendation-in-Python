# Movie-Recommendation-in-Python
Built a Movie Recommendation System for MovieLens dataset using pyTorch


Data Prep workflow:

After loading the data, I dropped timestamp since it was not needed. we removed the 18 duplicate records.

Created Xij matrix with the following approach:

Created the UserXMovies with pivot function in pandas and converted this to numpy arrays.
Xij is by the dot product of [MovieXUser].[UserXMatrix] producing [MovieXMovie] Matrix.

Model Building:

Built a linear model to learn the weights for the movie vectors.
Built model to learn the weights using gradient descent to find movie embeddings using Pytorch.
Created a linear model with three optimizers Stochastic Gradient Descent, Adagrad, Adam with learning rates 0.000001,0.01 and random the model with 1500 iterations.

--> Recommondation strategy - Calculated piecewise cosine similarity for the movie embeddings and found the top 10 movies with highest similarity value.
--> Found that the learning rate with an optimizer which converges quicker in less number of iterations.

--> Ran this for all the combinations of the optimizers and learning rates and found that the recommendations are different. 
--> The difference in the loss value is because of the standard learning rates across different optimizers. 
--> Re-ran adam optimizer with a higher learning rate and found recommendations most similar to our best model from above which is Stochastic Gradient Descent. 
