import predictionTimeSeriesModels as pstm
import createTimeSeriesData
import predictionModels as pred
import visualisations as vis

channelName = "Dude Perfect"

# invoke Time Series Predictions
data = createTimeSeriesData.getDateWiseGrouped(channelName)
pstm.performPredictions(data, channelName)

# invoke like/dislike ratio predictions
pred.performPredictions(channelName)

# perform vizualizations on time series analysis
vis.performVisualisations(channelName)
