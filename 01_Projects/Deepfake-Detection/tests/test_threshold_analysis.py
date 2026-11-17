import copy,unittest
from baselines.common.threshold_analysis import checked_rows,fit_validation_threshold,apply_threshold,summarize


def row(vid,label,score,split='validation'):
    return {'video_id':vid,'label':label,'score':score,'split':split,
        'prediction':int(score>=.5) if score is not None else None,
        'status':'ASSESSED' if score is not None else 'UNASSESSED'}


class ThresholdAnalysisTests(unittest.TestCase):
    def test_low_scores_are_calibrated_using_validation_only(self):
        rows=[row('a',0,.01),row('b',0,.02),row('c',1,.03),row('d',1,.04)]
        fitted=fit_validation_threshold(rows)
        self.assertEqual(fitted['threshold'],.03)
        self.assertEqual(fitted['selected_validation']['balanced_accuracy'],1)
        with self.assertRaisesRegex(ValueError,'validation'):
            fit_validation_threshold([{**r,'split':'test'} for r in rows])

    def test_denominator_and_scores_are_preserved_and_inputs_immutable(self):
        rows=[row('a',0,.1),row('b',1,.2),row('c',1,None)]
        before=copy.deepcopy(rows);fitted=fit_validation_threshold(rows)
        revised=apply_threshold(rows,fitted['threshold'])
        self.assertEqual(rows,before)
        self.assertEqual([r['score'] for r in rows],[r['score'] for r in revised])
        self.assertIsNone(revised[2]['prediction'])
        self.assertEqual(summarize(revised)['coverage'],2/3)

    def test_tied_scores_and_default_tie_policy(self):
        rows=[row('a',0,.2),row('b',1,.2)]
        self.assertEqual(fit_validation_threshold(rows)['threshold'],.5)

    def test_bad_predictions_fail_closed(self):
        for score in (float('nan'),float('inf'),-1,2):
            with self.assertRaises(ValueError):checked_rows([row('x',0,score)],'validation')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            checked_rows([row('x',0,.1),row('x',1,.2)],'validation')
        with self.assertRaisesRegex(ValueError,'both classes'):fit_validation_threshold([row('x',1,.1)])


if __name__=='__main__':unittest.main()
