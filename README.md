# pr_curves_extractor
PRCurvesExtractor class is used for Precision, Recall and PRcurves calculations. 

## Add prediction results
Prediction results that formatted in object_prediction proto are sorted and added to extractor by |AddObjPreds|.

### Sorting
Prediction results are sorted firstly by timestamps and then confidence scores using function |GetSortedIndexesByTimestampAndMergedScores|.

### Object processing and Matching strategy
* Processing object

The information of each object is proceessed and saved in BucketEvalInfo struct. In BucketEvalInfo:

**|bucket_prediction_info|** : score based information in vector of |AggregatedPredictionInfo|.

**|predicted_pr_info|** : |pred_class| and |label_class| based information in |AggregatedPredictionInfo|.

**|error_type_counts|** : the number of errors for FN and FP error types.

**|total_pos_label|** : label number. For classification, |total_pos_label| means number of true positives + number of false negative labels. Note that in classification tasks, positive predictions that match the same positive label are regarded as two true positives. For detection, |total_pos_label| simply means number of positive labels since one positive label matches at most one prediction as true positive.

* Matching strategy

Flags |prediction_is_positive|, |label_is_positive|, |has_enough_overlap|, |is_best_match|, |is_true_positive| are calculated to classify predictions and labels into TP, FP, TN, FN.

**|prediction_is_positive|**: If predicted class is positive.

**|label_is_positive|**: If associated label is positive.

**|is_best_match|**: Set to true for posotive labels that if the label id is not met before or previously matched predictions have no enough overlap. Note that for those predictions that |prediction_is_positive|=false are also able to match positive labels. For classification,
**|is_best_match|**: Set to true as long as |label_is_positive|=true.

**|is_true_positive|**: |prediction_is_positive| && |is_best_match|. **IMPORTANT**: |prediction_is_positive| && |label_is_positive| does not mean |is_true_positive|.

## Calculate Precision and Recall
Precision and recall for both classification and detection are calculated from |predicted_pr_info| by fomulas:

**Recall = TP / |total_pos_label|** and **Precision = TP / (TP + FP)**.

## Calculate PR curves
PR curves are calculated from |bucket_prediction_info|. Cumulative TP and FP are used.

**Recall(i)=cumulative TP(i)/total_pos_label**

**Predision(i)=cumulative TP(i)/(cumulative TP(i) + cumulative FP(i))**

* For detection, the only difference is that we directly use |total_pos_label| as it is, since one label can have at most one best match prediction (redundent predictions are trated as |FPNotBestMatch| error).

* For classification, in order to deal with the cases that multiple prediction associate with one label object, we define |total_pos_label|=number of TP predictions + number of FN labels. 
