# pr_curves_extractor
PRCurvesExtractor class is used for Precision, Recall and PRcurves calculations. 

## Add prediction results
All precition results that formatted in object_prediction proto are added to extractor by |AddObjPreds|.

### Sorting
Prediction results are sorted firstly by timestamps and then confidence scores using function |GetSortedIndexesByTimestampAndMergedScores|.

### Object processing and Matching strategy
* Processing object

The information of each object is proceessed and saved in BucketEvalInfo struct.
In BucketEvalInfo:
|bucket_prediction_info| stores score based information in vector of |AggregatedPredictionInfo|.
|predicted_pr_info| stores |pred_class| and |label_class| based information in |AggregatedPredictionInfo|.
|error_type_counts| counts the number of errors for FN and FP error types.
|total_pos_label| stores label number. For classification, |total_pos_label| means number of true positives + number of false negative labels. Note that in classification tasks, positive predictions that match the same positive label are regarded as two true positives. For detection, |total_pos_label| simply means number of positive labels since one positive label matches at most one prediction as true positive.

* Matching strategy

Flags |prediction_is_positive|, |label_is_positive|, |has_enough_overlap|, |is_best_match|, |is_true_positive| are calculated to classify predictions and labels into TP, FP, TN, FN.


## Precision and Recall
### Classification
### Detection
## PR curves
### Classification
### Detection
