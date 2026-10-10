"""Run: python -m unittest -v test_project.py. No API key or paid calls."""
import copy
import csv
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from schemas import Analysis, Evidence, Issue, QualityCheck, Sentiment
from helpers import load_conversations, load_criteria, verify_analysis, score_checks, report_csv
from analyzer import analyze, ai_analysis, AnalysisError

ROOT=Path(__file__).parent

class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.criteria=load_criteria((ROOT/'criteria.json').read_text())
        self.call=load_conversations(b'Agent: Hello!\nCustomer: My refund is missing.\nAgent: I will investigate.','call.txt')[0]
    def blank(self):
        return Analysis(issues=[],quality_checks=[],sentiment=[],risk_flags=[])
    def test_all_85_samples_and_offline_reports(self):
        calls=load_conversations((ROOT/'samples/mixed_dataset.json').read_bytes(),'mixed.json')
        self.assertEqual(len(calls),85)
        for c in calls:
            r=analyze(c,self.criteria)
            self.assertFalse(r['rejected_findings'])
            self.assertIsNone(r['metrics']['score'])
            self.assertEqual(r['mode'],'demo')
            json.loads(json.dumps(r))
    def test_uploaded_abcd_schema_and_actions(self):
        d=[{'convo_id':7,'original':[['agent','Hello'],['action','Account retrieved'],['customer','Need help']],'delexed':[{'irrelevant':True}]}]
        c=load_conversations(json.dumps(d).encode(),'abcd.json')[0]
        self.assertEqual(c.turns[1].speaker,'system');self.assertEqual(c.conversation_id,'7')
    def test_null_text_marks_gap(self):
        d={'turns':[{'speaker':'customer','text':'Help'},{'speaker':'agent','text':None}]}
        c=load_conversations(json.dumps(d).encode(),'call.json')[0]
        self.assertEqual(c.coverage,'incomplete');self.assertEqual(len(c.turns),2)
    def test_unknown_speaker_not_guessed(self):
        c=load_conversations(b'A statement without a speaker','x.txt')[0]
        self.assertEqual(c.turns[0].speaker,'unknown')
    def test_invalid_evidence_is_withheld(self):
        a=self.blank();a.issues=[Issue(title='Refund completed',outcome='resolved',explanation='Unsupported',evidence=[Evidence(turn_id='t003',quote='Your refund is complete')])]
        r,w,bad=verify_analysis(a,self.call,self.criteria)
        self.assertFalse(r.issues);self.assertEqual(len(bad),1)
    def test_absence_requires_confirmed_completeness(self):
        a=self.blank();a.quality_checks=[QualityCheck(criterion_id='closing',status='fail',explanation='No closing observed',support_type='absence',evidence=[])]
        r,_,_=verify_analysis(a,self.call,self.criteria)
        self.assertEqual(r.quality_checks[-1].status,'unknown')
        self.call.coverage='complete';r,_,_=verify_analysis(a,self.call,self.criteria)
        self.assertEqual(r.quality_checks[-1].status,'fail')
    def test_wrong_speaker_sentiment_withheld(self):
        a=self.blank();a.sentiment=[Sentiment(turn_id='t001',label='positive',explanation='Hello',evidence=[Evidence(turn_id='t001',quote='Hello!')])]
        r,_,_=verify_analysis(a,self.call,self.criteria);self.assertFalse(r.sentiment)
    def test_partial_score_excludes_unknown(self):
        a=self.blank();a.quality_checks=[QualityCheck(criterion_id='greeting',status='pass',explanation='Greeting',support_type='quote',evidence=[Evidence(turn_id='t001',quote='Hello!')])]
        r,_,_=verify_analysis(a,self.call,self.criteria);score=score_checks(r,self.criteria)
        self.assertEqual(score['score'],100);self.assertEqual(score['assessment_coverage_percent'],20)
    def test_duplicate_criterion_unknown(self):
        a=self.blank();q=QualityCheck(criterion_id='greeting',status='pass',explanation='Greeting',support_type='quote',evidence=[Evidence(turn_id='t001',quote='Hello!')]);a.quality_checks=[q,q]
        r,_,_=verify_analysis(a,self.call,self.criteria);self.assertEqual(r.quality_checks[0].status,'unknown')
    def test_csv_round_trip(self):
        buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=['conversation_id','transcript']);w.writeheader();w.writerow({'conversation_id':'a','transcript':'Agent: Hi, there\nCustomer: "Refund", please'})
        c=load_conversations(buf.getvalue().encode(),'x.csv')[0];self.assertEqual(c.turns[1].text,'"Refund", please')
    def test_empty_and_duplicate_inputs_rejected(self):
        for payload in [b'[]',b'{bad',b'[{"conversation_id":"x","transcript":"Customer: Help"},{"conversation_id":"x","transcript":"Customer: Help"}]']:
            with self.assertRaises(ValueError):load_conversations(payload,'x.json')
    @patch('analyzer.requests.post')
    def test_api_contract_and_no_label_leakage(self,post):
        response=Mock(status_code=200);response.json.return_value={'choices':[{'finish_reason':'stop','message':{'content':self.blank().model_dump_json()}}]};post.return_value=response
        ai_analysis(self.call,self.criteria,'not-a-real-key','gpt-4.1-mini')
        body=post.call_args.kwargs['json'];data=json.loads(body['messages'][1]['content'])
        self.assertEqual(set(data),{'coverage','criteria','turns'})
        self.assertEqual(body['response_format']['type'],'json_schema')
        self.assertNotIn('expected_checks',json.dumps(data))
    @patch('analyzer.requests.post')
    def test_api_auth_error_is_not_fake_success(self,post):
        post.return_value=Mock(status_code=401)
        with self.assertRaisesRegex(AnalysisError,'Invalid API key'):ai_analysis(self.call,self.criteria,'fake','gpt-4.1-mini')
    @patch('analyzer.requests.post')
    def test_truncated_response_rejected(self,post):
        response=Mock(status_code=200);response.json.return_value={'choices':[{'finish_reason':'length','message':{'content':'{}'}}]};post.return_value=response
        with self.assertRaisesRegex(AnalysisError,'incomplete'):ai_analysis(self.call,self.criteria,'fake','gpt-4.1-mini')
    def test_report_csv_formula_escape_and_review(self):
        r=analyze(self.call,self.criteria);r['reviews']['issues:0']={'decision':'accept','notes':'=1+1'}
        rows=list(csv.DictReader(io.StringIO(report_csv(r).decode('utf-8-sig'))))
        row=next(x for x in rows if x['finding_id']=='issues:0');self.assertEqual(row['review_notes'],"'=1+1");self.assertEqual(row['review_decision'],'accept')

if __name__=='__main__':unittest.main()
