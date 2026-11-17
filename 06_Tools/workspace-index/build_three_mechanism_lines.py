"""Publication line figures using local measured workloads and retained scores only."""
import csv, hashlib, html, json, os, sys, zipfile
from pathlib import Path
W=next(p for p in Path(__file__).resolve().parents if (p/'workspace-layout.json').is_file())
os.environ['MPLCONFIGDIR']=str(W/'09_Runtime/caches/main/matplotlib')
from journal_visuals_common import Pack, Sources, setup_style
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FixedLocator, ScalarFormatter, NullFormatter
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, f1_score
R=W/'04_Results/experiments/fk'; B=R/'training-scaling-001'
O=W/'04_Results/publication/three-mechanisms-lines-001'
METHODS=['Ours','AVFakeNet','AVTENet']; COLORS=['#1976b7','#e18424','#cb3e48']; MARKERS=['o','s','D']
STAGES={'Ours':['ours'],'AVFakeNet':['avf-visual','avf-audio'],'AVTENet':['avt-audio','avt-visual','avt-audiovisual','avt-fusion']}
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d): p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''): h.update(x)
    return h.hexdigest()
def csv_out(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        c=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));c.writeheader();c.writerows(rows)
def grid(ax,work=False):
    ax.set_axisbelow(True);ax.grid(which='major',color='#b9bec3',alpha=.65,linewidth=.45)
    ax.grid(which='minor',color='#d3d6d9',alpha=.55,linewidth=.3)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if work:
        ax.set_xscale('log',base=2);ax.set_xlim(.88,9.2);ax.xaxis.set_major_locator(FixedLocator([1,2,4,8]));ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(FixedLocator([1.25,1.5,1.75,2.5,3,3.5,5,6,7]));ax.xaxis.set_minor_formatter(NullFormatter());ax.set_ylim(bottom=0)
        ax.set_xlabel('Number of 2-second window presentations')
    else:
        ax.set_xlim(0,1);ax.set_ylim(0,1.025);ax.set_xticks(np.arange(0,1.01,.2));ax.set_yticks(np.arange(0,1.01,.2));ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate')
    for sp in ax.spines.values():sp.set_color('#656c73');sp.set_linewidth(.6)

class Study:
    def __init__(self):
        O.mkdir(parents=True,exist_ok=True);setup_style();plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'lines.linewidth':1.45,'lines.markersize':5})
        self.sources=Sources();self.pack=Pack(O,self.sources);self.audit={}
    def source(self,p,group):self.sources.register(p,group);return p
    def json(self,p,g):return read(self.source(p,g))
    def figure(self,slug,title,caption,fig,rows,group,scope):
        self.pack.figure(slug,title,caption,fig,rows,[group],section='Measured line comparisons')
        it=self.pack.items[-1];it.update(label='Fig. M'+str(it['number']),source_files=sorted(self.sources.groups[group]),comparison_scope=scope)
        write(O/'metadata'/(it['stem']+'.json'),it)
    def workload(self,extract=False):
        group='extraction' if extract else 'training'; records=[];stage_rows=[];curves={};staged={}
        self.json(B/'protocol.json',group)
        for method,stages in STAGES.items():
            stages=stages[:-1] if extract and method=='AVTENet' else stages
            parts=[]
            for stage in stages:
                raw=self.json(B/(('extract-' if extract else '')+stage+'.json'),group)
                assert raw['checkpoint_unchanged']
                data=raw['curves'];data=data['online'] if isinstance(data,dict) else data
                assert [x['window_presentations'] for x in data]==[1,2,4,8]
                parts.append(data)
                for point in data:
                    for sample in point['samples']:
                        stage_rows.append({'method':method,'stage':stage,'window_presentations':point['window_presentations'],**sample})
            ys=[]
            for i,n in enumerate([1,2,4,8]):
                samples=[sum(p[i]['samples'][r]['seconds'] for p in parts) for r in range(3)]
                median=float(np.median(samples));ys.append(median)
                updates=0 if extract else sum(p[i]['updates_per_presentation']*n for p in parts)
                records.append({'method':method,'window_presentations':n,'source_seconds_presented':2*n,'seconds':median,'repeat_0_seconds':samples[0],'repeat_1_seconds':samples[1],'repeat_2_seconds':samples[2],'optimizer_updates':updates,'aggregation':'median of three stage-summed replays','panel':'a'})
            curves[method]=ys;staged[method]=stages
        fig,ax=plt.subplots(figsize=(6.7,4.4))
        for m,c,marker in zip(METHODS,COLORS,MARKERS):ax.plot([1,2,4,8],curves[m],color=c,marker=marker,label=m)
        grid(ax,True);ax.set_ylabel('Summed stage time (s)');ax.set_title('Feature extraction workload' if extract else 'Training workload comparison')
        ax.legend(loc='upper left',framealpha=.97,edgecolor='#aeb4b9');ax.set_ylim(0,max(curves['AVTENet'])*1.13)
        for m,c in zip(METHODS,COLORS):ax.annotate(f'{curves[m][-1]:.3f} s',(8,curves[m][-1]),xytext=(-5,8),textcoords='offset points',ha='right',fontsize=8,color=c)
        if not extract:
            raw=self.json(B/'ours.json',group)['curves']['cached'];cached=[p['median_seconds'] for p in raw]
            # A distinct inset is explicitly an internal comparison; it does not replace online Ours above.
            inset=ax.inset_axes([.08,.42,.42,.29]);inset.plot([1,2,4,8],curves['Ours'],color=COLORS[0],marker='o',markersize=3,label='Ours: online')
            inset.plot([1,2,4,8],cached,color='#7753a3',marker='^',markersize=3,ls='--',label='Ours: cached')
            inset.set_xticks([1,2,4,8]);inset.set_ylim(0,.61);inset.set_title('Our cache comparison',fontsize=8);inset.set_ylabel('Seconds',fontsize=7);inset.tick_params(labelsize=7);inset.grid(alpha=.3);inset.legend(fontsize=6,loc='upper left')
            for p in raw:records.append({'method':'Ours: cached','window_presentations':p['window_presentations'],'source_seconds_presented':p['window_presentations']*2,'seconds':p['median_seconds'],**{'repeat_'+str(s['repeat'])+'_seconds':s['seconds'] for s in p['samples']},'optimizer_updates':p['window_presentations'],'aggregation':'median of three replays','panel':'b'})
        csv_out(O/'data'/(group+'-stage-replays.csv'),stage_rows)
        self.audit[group]={'curves_seconds':curves,'stages':staged,'points_per_series':4,'repeats_per_stage':3}
        if extract:
            caption='Same prepared 2-second TRAIN windows; 1, 2, 4 and 8 presentations, cycled from two overlapping genuine windows. Points are medians of three sums of separately timed native stages; lines connect observations. Ours includes frozen encoders, GRU, bin pooling and projections. AVFakeNet sums visual and audio DST forward paths, including the classifier exposed by its native API. AVTENet sums AST, ViViT and AV-HuBERT/TCN feature extraction; fusion is excluded. Preparation and model loading are excluded; native feature-return transfers are included. This measures runtime, not intrinsic representation or localization quality.'
            self.figure('feature-evidence-extraction','Feature extraction: measured workload scaling',caption,fig,records,group,{'source':'two fixed TRAIN windows','comparison':'native feature paths','x':'window presentations','y':'seconds; median of stage sums'})
        else:
            caption='Same prepared 2-second TRAIN windows; 1, 2, 4 and 8 presentations, cycled from two overlapping genuine windows. Points are medians of three sums of separately timed stages; lines connect observations. Main Ours includes online frozen encoding and head training. AVFakeNet includes both branches; AVTENet includes all three expert updates plus cached fusion. Per presentation: Ours 1, AVFakeNet 8 and AVTENet 4 optimizer updates. Preparation, initialization, disk/cache I/O and expert-to-fusion cache creation are excluded. Inset compares our online and precomputed-feature paths only. Updates are discarded. This is a short workload probe, not full-epoch or convergence timing.'
            self.figure('training-workload','Training: measured workload scaling',caption,fig,records,group,{'source':'two fixed TRAIN windows','comparison':'all trained stages with native update counts','x':'window presentations','y':'seconds; median of stage sums','internal':'Ours online versus cached inset'})
    def scoring(self):
        g='scoring';p=self.source(R/'paper-style-visuals-001/data/fig05-roc-pr.csv',g)
        with p.open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
        fig,axs=plt.subplots(1,2,figsize=(9,4));rows=[]
        names={'FK':'Ours','AVFakeNet adaptation':'AVFakeNet','AVTENet adaptation (0.5)':'AVTENet'}
        for m,c,marker in zip(METHODS,COLORS,MARKERS):
            part=[r for r in raw if names[r['method']]==m and r['curve'] in ('ROC','hard ROC point')]
            x=[float(r['x']) for r in part];y=[float(r['y']) for r in part]
            if m=='AVFakeNet':axs[0].scatter(x,y,label=m+' (decision only)',color=c,marker=marker,s=30,zorder=5)
            else:axs[0].plot(x,y,label=m,color=c,marker={'Ours':'o','AVTENet':'^'}[m],markevery=.12,markersize=4.5,markerfacecolor='white',markeredgewidth=.9)
            rows.extend({'panel':'a','method':m,'curve':'hard ROC point' if m=='AVFakeNet' else 'ROC','x':a,'y':b,'assessed_clips':415} for a,b in zip(x,y))
        variants={v:self.json(R/'remaining-ablation-evaluation-001'/v/'test.json',g) for v in ('full','uniform_weights')}
        indexed={v:{r['video_id']:r for r in d['videos'] if r['prediction'] is not None and r['score'] is not None} for v,d in variants.items()}
        ids=sorted(set.intersection(*(set(d) for d in indexed.values())));assert len(ids)==980
        assert all(indexed['full'][i]['label']==indexed['uniform_weights'][i]['label'] for i in ids)
        metrics={}
        for v,c,marker,label in [('full',COLORS[0],'o','Ours: learned weights'),('uniform_weights','#7753a3','^','Ours: uniform weights')]:
            rr=[indexed[v][i] for i in ids];y=[r['label'] for r in rr];score=[r['score'] for r in rr];pred=[r['prediction'] for r in rr]
            x,tpr,_=roc_curve(y,score);auc=roc_auc_score(y,score);metrics[v]={'auc':float(auc),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'f1':float(f1_score(y,pred))}
            axs[1].plot(x,tpr,label=label,color=c,marker=marker,markersize=4.5,markevery=.12,markerfacecolor='white',markeredgewidth=.9)
            rows.extend({'panel':'b','method':label,'curve':'ROC','x':float(a),'y':float(b),'assessed_clips':980} for a,b in zip(x,tpr))
        for ax in axs:
            ax.plot([0,1],[0,1],ls=':',color='#929aa0',lw=.8);grid(ax);ax.legend(loc='lower right',fontsize=7)
        axs[0].set_title('(a) Model comparison: same 415 clips');axs[1].set_title('(b) Our scoring control: same 980 clips')
        self.audit[g]={'internal_metrics':metrics,'external_assessed_clips':415,'internal_assessed_clips':980,'avfakenet_continuous_scores_available':False}
        self.figure('scoring-effectiveness','Scoring: ROC comparison and weighting control',
            '(a) Retained scores on the same 415 assessed clips: Ours and AVTENet have continuous ROC curves. AVFakeNet supplies hard decisions only, so its measured operating point is shown without an invented curve or AUC. (b) Internal comparison of our learned and uniform weighting on the same 980 assessed clips, from the retained single-seed ablation. These panels use separate cohorts. The internal control supports weighting in this setting; the external comparison includes differences throughout each model. Other retained aggregation rules can improve some metrics; the complete design is not universally optimal.',
            fig,rows,g,{'external':'415 common assessed IDs','internal':'980 matched weighting-control IDs','x':'false positive rate','y':'true positive rate'})
    def finish(self):
        sys.path.insert(0,str(W/'09_Runtime/review-cache/archive-v0/1PiHw63lG2QUPDDZ'));import pymupdf as fitz
        doc=fitz.open();qa=O/'qa';qa.mkdir(exist_ok=True)
        for item in self.pack.items:
            page=doc.new_page(width=960,height=660)
            page.insert_text((35,38),item['label']+'. '+item['title'],fontname='tibo',fontsize=18,color=(.03,.24,.38))
            pix=fitz.Pixmap(str(O/item['png']));scale=min(885/pix.width,435/pix.height);ww,hh=pix.width*scale,pix.height*scale;xx=(960-ww)/2
            page.insert_image(fitz.Rect(xx,62,xx+ww,62+hh),pixmap=pix)
            assert page.insert_textbox(fitz.Rect(35,515,925,627),item['caption'],fontname='tiro',fontsize=10,lineheight=1.2)>=0
            page.insert_text((35,645),'Own measured data | Short workload probes; selected checkpoints unchanged',fontsize=8,fontname='tiro')
        doc.save(O/'THREE_MECHANISMS.pdf',garbage=4,deflate=True)
        for i,page in enumerate(doc):page.get_pixmap(dpi=110).save(qa/f'mechanism-{i+1:02d}.png')
        doc.close()
        gallery='<html lang="en"><meta charset="utf-8"><title>Three mechanism line comparisons</title><style>body{max-width:1150px;margin:30px auto;padding:0 20px;font:17px Georgia;line-height:1.5;color:#182733}img{max-width:100%}article{border-top:1px solid #ccc;margin-top:24px}h1{font-size:28px}h2{font-size:22px}</style><h1>Training, scoring and extraction: line comparisons</h1><p><a href="THREE_MECHANISMS.pdf">PDF</a> | <a href="THREE_MECHANISMS_PNG.zip">3 PNGs</a> | <a href="FINDINGS_TH.md">Thai interpretation</a> | <a href="LIMITATIONS.md">Limitations</a></p>'
        for item in self.pack.items:gallery+='<article><h2>'+html.escape(item['label']+'. '+item['title'])+'</h2><img src="'+item['png']+'"><p>'+html.escape(item['caption'])+'</p><a href="'+item['svg']+'">SVG</a> | <a href="'+item['data']+'">CSV</a></article>'
        (O/'index.html').write_text(gallery+'</html>',encoding='utf-8')
        write(O/'catalog.json',{'items':self.pack.items});write(O/'measurements.json',self.audit)
        write(O/'source-manifest.json',{'files':self.sources.files,'groups':{g:sorted(v) for g,v in self.sources.groups.items()},'origin':'own measured experiments only'})
        with zipfile.ZipFile(O/'THREE_MECHANISMS_PNG.zip','w',zipfile.ZIP_DEFLATED) as z:
            for item in self.pack.items:z.write(O/item['png'],Path(item['png']).name)
        tr=self.audit['training']['curves_seconds'];ex=self.audit['extraction']['curves_seconds'];metrics=self.audit['scoring']['internal_metrics']
        (O/'FINDINGS_TH.md').write_text(f'''# กราฟเส้น 3 กลไก

ทุกจุดใช้ผลวัดของเราเอง เส้นเชื่อมจุดช่วยอ่านแนวโน้ม ไม่มีการสร้างจุดวัดหรือใช้ค่าจาก paper ตัวอย่าง

## M1: Training workload

แกน X คือจำนวนครั้งที่ป้อนหน้าต่างวิดีโอ 2 วินาที (1, 2, 4, 8 ครั้ง; สเกล log ฐาน 2) แกน Y คือเวลารวม stage หน่วยวินาที ค่ายิ่งต่ำยิ่งใช้เวลาน้อย

ที่ 8 ครั้ง: Ours {tr['Ours'][-1]:.3f} s, AVFakeNet {tr['AVFakeNet'][-1]:.3f} s, AVTENet {tr['AVTENet'][-1]:.3f} s กราฟหลักรวม online encoding ของ Ours ด้วย ช่องเล็กเป็นการเทียบ Ours online กับ cached ของเราเอง

ทุกวิธีใช้ต้นทางหน้าต่างเดียวกัน แต่จำนวน optimizer updates ต่างกัน: Ours 1, AVFakeNet 8 และ AVTENet 4 ต่อครั้งที่ป้อน หน้าต่างมาจากวิดีโอ TRAIN จริงหนึ่งคลิป เริ่มที่ 0 และ 1 วินาที จึงซ้อนทับกัน และวนป้อนซ้ำ นี่คือ microbenchmark ภาระงาน ไม่ใช่เวลาเทรนครบ epoch หรือหลักฐานว่า convergence ดีกว่า

กลไกที่สอดคล้องกับการลดงานคือ freeze encoders และ cache features ยังไม่ได้แยกผลเชิงเหตุของ pretraining, filter หรือ normalization

## M2: Scoring effectiveness

แกน X คือ false positive rate แกน Y คือ true positive rate เส้นที่เข้าใกล้มุมซ้ายบนแยกคลาสได้ดีกว่า ช่องซ้ายเปรียบเทียบกับ baseline บน 415 คลิปเดียวกัน AVFakeNet มีเพียง hard decisions จึงแสดงได้เพียงจุดจริง ไม่สร้างเส้น ROC ให้

ช่องขวาเป็นการเทียบกระบวนการของเราเอง: learned weights กับ uniform weights บน 980 คลิปเดียวกัน AUC เท่ากับ {metrics['full']['auc']:.4f} และ {metrics['uniform_weights']['auc']:.4f} ตามลำดับ BA เพิ่ม {(metrics['full']['balanced_accuracy']-metrics['uniform_weights']['balanced_accuracy'])*100:.2f} จุดเปอร์เซ็นต์ และ F1 เพิ่ม {(metrics['full']['f1']-metrics['uniform_weights']['f1'])*100:.2f} จุดเปอร์เซ็นต์

สองช่องใช้คนละ cohort ห้ามรวมเป็นชุดเดียว ผลเทียบ baseline ไม่ได้แยกผลของ scoring เพียงส่วนเดียว และ aggregation แบบอื่นที่เก็บไว้เดิมอาจได้บาง metric สูงกว่า Full

## M3: Feature extraction workload

แกน X เหมือน M1 แกน Y คือเวลารวม native feature paths ที่ 8 ครั้ง: Ours {ex['Ours'][-1]:.3f} s, AVFakeNet {ex['AVFakeNet'][-1]:.3f} s, AVTENet {ex['AVTENet'][-1]:.3f} s

Ours รวม encoders, GRU, bin pooling และ projections; AVFakeNet รวม DST visual/audio ซึ่ง API รวม classifier ไว้ด้วย; AVTENet รวม AST, ViViT, AV-HuBERT/TCN โดยไม่รวม fusion ไม่รวม preprocessing และการโหลดโมเดล จึงเปรียบเทียบเวลาของ feature paths ตาม implementation ไม่ใช่พิสูจน์คุณภาพ representation หรือ evidence

ผลชุดนี้เป็น fixed-window benchmark ใหม่ ขอบเขตต่างจากการวัดทั้งคลิป 5 คลิปเดิม ซึ่ง Ours ช้ากว่า AVFakeNet เล็กน้อยใน extraction; ไม่ใช้ผลใหม่นี้แทนข้อสรุปเดิมโดยไม่อธิบาย protocol

วัดซ้ำ stage ละ 3 ครั้งบน RTX 3070 แล้วหาค่ามัธยฐานของผลรวม stage ที่วัดแยกกัน ไม่ใช่จับเวลารันทั้งระบบต่อเนื่อง น้ำหนัก checkpoint ที่เลือกไว้ไม่ได้ถูกแก้ไข
''',encoding='utf-8')
        (O/'LIMITATIONS.md').write_text('''# Measurement and interpretation limits

- Labels AVFakeNet and AVTENet refer to our local adaptations, not author-reported results or certified official reproductions. No numerical values from the example papers are used.
- Workload probes cycle two overlapping genuine TRAIN windows from one video. N is the number of presentations, not the number of unique clips. Three repeats quantify short runtime observations; they do not establish population scaling, multi-seed variability, full-epoch time, convergence, or accuracy after these updates.
- Equal source-media presentations do not imply equal optimizer steps. Per presentation, Ours has one evidence-head update, AVFakeNet has seven visual minibatch updates plus one audio update, and AVTENet has three expert updates plus one cached fusion update. AVTENet expert-to-fusion cache formation is excluded. Stage loading, preparation, optimizer initialization and disk/cache I/O are excluded throughout.
- Each baseline stage is timed separately on the same RTX 3070. Each plotted point is the median of three stage sums, not a direct end-to-end ensemble timer. The Ours online main curve includes frozen encoding; the cached inset excludes feature generation. Warmup occurs outside timing. Native optimizers and losses differ. FP32/TF32-off is fixed, but Windows torch 2.2.2 and Linux torch 1.13.1 runtime differences remain.
- Extraction boundaries differ: Ours includes encoders, GRU, bin pooling and feature projections; AVFakeNet native forward exposes classifier outputs and includes those small heads; AVTENet returns AST/ViViT/AV-HuBERT-TCN features without fusion. Native return transfers are included. These are measured implementation feature paths, not identical kernels or intrinsic feature/evidence quality. Fixed-window results are separate from the earlier five-whole-clip profiling protocol.
- Scoring external curves use 415 common assessed IDs. AVFakeNet provides no retained continuous scores and has only a hard ROC point. The internal learned/uniform comparison uses 980 matched assessed IDs and one retained training seed. Model-level differences do not isolate scoring. Other retained aggregation rules improve some metrics; Full is not universally optimal.
- No controlled ablation isolates pretraining, filtering or normalization. Baselines also use pretrained components. Cost differences cannot be attributed uniquely to any of these mechanisms.
- Native temporal localization, signed bin contributions and source-bound records are available in our evaluated implementation; availability is not a quality score or an impossibility claim about baseline methods. Record integrity is not media authenticity; signed contributions are not causal explanations. Existing localization assessed-region F1 is 0.469.
- Training updates were discarded and protected source checkpoints remained unchanged. No benchmark updates were used for scoring curves or model selection. Lines connect measured points only; no extrapolated values or confidence intervals are shown.
''',encoding='utf-8')
        assert all(sha(W/p)==entry['sha256'] for p,entry in self.sources.files.items())
        write(O/'validation.json',{'status':'GENERATED','figures':3,'source_files_unchanged':True,'new_training_benchmark':True,'new_extraction_benchmark':True,'selected_checkpoints_unchanged':True,'reference_paper_data_used':False,'manual_visual_review':'pending'})
        print(json.dumps(self.audit),flush=True)

if __name__=='__main__':
    study=Study();study.workload();study.scoring();study.workload(True);study.finish()
