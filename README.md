# PRCurvesExtractor
|PRCurvesExtractor| class is used for precision, recall, pr curves calculations. 
It is implemented in pr_curves_extractor.h/.cpp.

## Internal states:

**|pos_cls_combines_|:** |pos_cls_combines_| describes all the combinations used for PR results in this extractor. Each element is a combination of classes, the classes inside the combiatnion will be merged.

**|curr_label_to_matched_prediction|:** Used for storing positive labels and the associated |pred_obj| information.

**|cumulative_bucket_eval_info_|:** |BucketEvalInfo| from accumulative data of all the combines in |pos_cls_combines_|. |BucketEvalInfo| contains:

* **|bucket_prediction_info|** : Score based prediction information, which is used for  PR-curves cumulative calculation.

* **|predicted_pr_info|** : Predict label based prediction information, which is used for precision and recall calculation.

* **|error_type_counts|** : The number of errors for FN and FP error types.

* **|total_pos_label|** : Label number. For classification, |total_pos_label| means number of true positives + number of false negative labels. Note that in classification tasks, positive predictions that match the same positive label are regarded as two true positives. For detection, |total_pos_label| simply means number of positive labels since one positive label matches at most one prediction as true positive.

## Major public functions:

**AddObjPreds:** Accumulates |pred_results| into internal states. Output results are calculated from the internal states by calling |ExportPRCurves| or |ExportPRs|.

**ExportPRCurves:** Exports PR Curves for |pos_cls_combines_|. This is used for generating the data for ploting pr curves.

**ExportPRs:** Calculates the precisions and recalls for |pos_cls_combines_|. This is used for the precision and recall shown in PRT result tables and reports.

## Add prediction results
Prediction results that formatted in object_prediction proto are sorted and added to extractor by |AddObjPreds|. Details are introduced as follows in the flow as it works: 

Sorting --> Object processing and matching strategy --> Update FN

### Sorting
Prediction results are sorted firstly by timestamps and then confidence scores using function |GetSortedIndexesByTimestampAndMergedScores|.

### Object processing and matching strategy
* Processing object

The information of each object is proceessed and saved in BucketEvalInfo struct.

* Matching strategy

Flags |prediction_is_positive|, |label_is_positive|, |has_enough_overlap|, |is_best_match|, |is_true_positive| are calculated to classify predictions and labels into TP, FP, FN.

**|prediction_is_positive|**: If predicted class(|pred_class|) is positive.

**|label_is_positive|**: If associated label(|label_class|) is positive.

**|is_best_match|**: Set to true for posotive labels that if the label id is not met before or previously matched predictions have no enough overlap. Note that for those predictions that |prediction_is_positive| = false are also able to match positive labels. For classification, |is_best_match| is set to true as long as |label_is_positive| is true.

**|is_true_positive|**: |prediction_is_positive| && |is_best_match|. **IMPORTANT**: |prediction_is_positive| && |label_is_positive| does not mean |is_true_positive|. |obj_pred| should also have enough overlap with the associate label and the label has not been matched before.

TP : |is_true_positive|==true.

FP : |prediction_is_positive|==true && |is_true_positive|==false.

FN : For each |pred_obj|, if |label_is_positive| is true, check if the label is marked as best matched in |label_to_matched_prediction|. If no, update the label and the associate |pred_obj| that has largest overlap to |label_to_matched_prediction|. After all the |pred_obj| are processed for a frame, FN is calculated by |UpdateFN|, which is explained as follows.

### Update FN
|UpdateFN| updates all labels and adds positive labels that are not counted as true positive to false negatives. 

For each |label_prediction_pair| in |label_to_matched_prediction|, if the |is_pred_positive| is true and the prediction has enough overlap, the |label_prediction_pair| is regarded as TP and skip updating FN. Otherwise, increase |total_pos_label| by 1 and count type error for object detection task.

## Calculate Precision and Recall
Precision and recall for both classification and detection are calculated from |predicted_pr_info| by formulas:

**Recall = TP / |total_pos_label|** and **Precision = TP / (TP + FP)**.

## Calculate PR curves
PR curves are calculated from |bucket_prediction_info|. Cumulative TP and FP are used.

**Recall(i)=cumulative TP(i)/total_pos_label**

**Predision(i)=cumulative TP(i)/(cumulative TP(i) + cumulative FP(i))**

* For detection, the only difference is that we directly use |total_pos_label| as it is, since one label can have at most one best match prediction (redundent predictions are trated as |FPNotBestMatch| error).

* For classification, in order to deal with the cases that multiple prediction associate with one label object, we define |total_pos_label|=number of TP predictions + number of FN labels. 
