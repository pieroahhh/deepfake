"""Three evidence-backed mechanism figures from retained local experiments.

Training: trainable-weight inventory, not a cross-method full-epoch timing claim.
Scoring: external detection and matched internal scoring-weight ablation.
Extraction: real extraction time, native evidence outputs, own modality controls.
"""
import csv,hashlib,html,json,os,shutil,sys,zipfile
from pathlib import Path
W=next(p for p in Path(__file__).resolve().parents if (p/'workspace-layout.json').is_file())
os.environ['MPLCONFIGDIR']=str(W/'09_Runtime/caches/main/matplotlib')
sys.path.insert(0,str(W/'06_Tools/workspace'))
from workspace_paths import relocate
from journal_visuals_common import Pack,Sources,setup_style,axes_grid
import matplotlib.pyplot as plt
import numpy as np
R=W/'04_Results/experiments/fk'
OLD=R/'paper-style-visuals-001'
TRAIN=R/'table-iii-complete-001'
OUT=W/'04_Results/publication/three-mechanisms-001'
METHODS=['Ours','AVFakeNet','AVTENet']
COLORS=['#2378ad','#409b49','#d85143']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def method(value):
    if value=='FK':return 'Ours'
    if value.startswith('AVFakeNet'):return 'AVFakeNet'
    if value.startswith('AVTENet'):return 'AVTENet'
    return value
def export_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


class Study:
    def __init__(self):
        OUT.mkdir(parents=True,exist_ok=True);setup_style();self.sources=Sources();self.pack=Pack(OUT,self.sources);self.integrity={};self.audit={}
    def source(self,p,group):
        p=relocate(Path(p),__file__).resolve();self.integrity[str(p)]=sha(p);self.sources.register(p,group);return p
    def json(self,p,group):return read(self.source(p,group))
    def csv(self,p,group):
        with self.source(p,group).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    def figure(self,slug,title,caption,fig,rows,group,scope):
        self.pack.figure(slug,title,caption,fig,rows,[group],section='Three mechanisms')
        item=self.pack.items[-1];item['label']='Fig. M'+str(item['number']);item['source_files']=sorted(self.sources.groups[group]);item['comparison_scope']=scope
        write(OUT/'metadata'/(item['stem']+'.json'),item)
        return item

    def training(self):
        group='training'
        raw={s:self.json(TRAIN/s/'training-cost.json',group) for s in ('fk','avfakenet','audio','visual','audiovisual','fusion')}
        ident=self.json(R/'table-iii-cost-001/fk/identity.json',group)
        ours=raw['fk']['cached_training'][0]['benchmark']['parameters']['trainable']
        assert ours==ident['model']['parameters'] and ident['encoders']['requires_grad_parameters']==0
        stages=[{'method':'Ours','stage':'evidence head','trainable_parameters':ours,'frozen_encoder_parameters':ident['encoders']['parameters']}]
        for branch in ('visual','audio'):
            params=raw['avfakenet']['branches'][branch]['benchmark']['parameters']
            stages.append({'method':'AVFakeNet','stage':branch+' DST','trainable_parameters':params['trainable'],'frozen_encoder_parameters':0})
        for stage in ('audio','visual','audiovisual'):
            params=raw[stage]['experts'][0]['benchmark']['parameters']
            assert all(r['benchmark']['parameters']==params for r in raw[stage]['experts'])
            stages.append({'method':'AVTENet','stage':{'audio':'AST expert','visual':'ViViT expert','audiovisual':'AV-HuBERT + temporal expert'}[stage],'trainable_parameters':params['trainable'],'frozen_encoder_parameters':0})
        stages.append({'method':'AVTENet','stage':'cached fusion head','trainable_parameters':raw['fusion']['cached_training']['parameters']['trainable'],'frozen_encoder_parameters':0})
        totals={m:sum(s['trainable_parameters'] for s in stages if s['method']==m) for m in METHODS}
        assert totals=={'Ours':1002504,'AVFakeNet':13540732,'AVTENet':284662002}
        matched=raw['fk']['matched_online_cached'];online=matched['online']['medians']['total_seconds'];cached=matched['cached']['medians']['total_seconds'];speedup=online/cached
        assert matched['final_weight_max_absolute_error']==0 and max(matched['feature_max_absolute_errors'].values())==0
        fig,axs=plt.subplots(1,2,figsize=(8.7,3.65),gridspec_kw={'width_ratios':[1.1,1]})
        values=[totals[m]/1e6 for m in METHODS];bars=axs[0].bar(METHODS,values,color=COLORS,edgecolor='black',linewidth=.35)
        axs[0].set_yscale('log');axs[0].set_ylim(.5,800);axs[0].set_yticks([1,10,100],['1','10','100']);axs[0].set_ylabel('Trainable parameters (millions, log scale)');axs[0].set_title('(a) Trainable-weight inventory across stages')
        for bar,v in zip(bars,values):axs[0].annotate(f'{v:.3f} M',(bar.get_x()+bar.get_width()/2,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
        bars=axs[1].bar(['Ours: online','Ours: cached'],[online*1000,cached*1000],color=['#df927f',COLORS[0]],edgecolor='black',linewidth=.35)
        axs[1].set_ylim(0,470);axs[1].set_ylabel('Milliseconds per complete-video update');axs[1].set_title('(b) Our measured cache benefit')
        for bar,v in zip(bars,[online*1000,cached*1000]):axs[1].annotate(f'{v:.2f} ms',(bar.get_x()+bar.get_width()/2,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
        axs[1].text(.5,.9,f'{speedup:.2f}x in this prepared-input replay',transform=axs[1].transAxes,ha='center',fontsize=8)
        for ax in axs:axes_grid(ax)
        rows=[{'panel':'a','method':m,'measurement':'distinct_trainable_parameter_inventory','value':totals[m],'unit':'parameters','scope':'sum each distinct trained stage once; not simultaneous execution'} for m in METHODS]
        rows += [{'panel':'b','method':'Ours','measurement':mode+'_update','value':value,'unit':'seconds','scope':'one eight-window TRAIN video; prepared inputs; media preprocessing and cache I/O excluded'} for mode,value in [('online',online),('cached',cached)]]
        self.figure('training-footprint','Training: trainable-weight footprint and feature caching',
            '(a) Inventory of parameters designated trainable across the distinct stages of retained models/training probes: Ours trains its evidence head while 99.660 M encoder parameters stay frozen; AVFakeNet has two trained branches; AVTENet has three expert-training stages plus fusion. Each distinct stage is counted once. This is a gradient-update footprint proxy, not measured full-epoch time, actual peak gradient memory or total model size. AVTENet cached fusion alone has only 2,305 trainable parameters. (b) Same Ours model and eight-window TRAIN video; online/cached features and final weights match exactly. Media preparation and disk/cache I/O are excluded. No cross-method wall-time training speedup is claimed.',
            fig,rows,group,{'external_comparison':'trainable-parameter inventory','internal_comparison':'matched online versus cached training'})
        export_csv(OUT/'data/training-stage-inventory.csv',stages)
        self.audit['training']={'distinct_optimized_parameters':totals,'frozen_ours_encoder_parameters':ident['encoders']['parameters'],'ours_cached_speedup_in_scope':speedup,'baseline_walltime_superiority_proven':False}

    def scoring(self):
        group='scoring';common=self.csv(OLD/'tables/table03-main-common.csv',group)
        ablation=self.json(R/'remaining-ablation-evaluation-001/comparison/metrics.json',group)
        agg=self.csv(OLD/'tables/table14-aggregation.csv',group)
        metrics=ablation['common_assessed'];assert metrics['full']['assessed']==metrics['uniform_weights']['assessed']==980
        deltas={};rows=[]
        fig,axs=plt.subplots(1,2,figsize=(9.0,3.7),gridspec_kw={'width_ratios':[1.1,1]});x=np.arange(3);width=.33
        for j,(field,label,color) in enumerate([('BA %','Balanced accuracy','#2378ad'),('F1 %','F1 score','#e5a754')]):
            values=[float(next(r[field] for r in common if method(r['Method'])==m)) for m in METHODS]
            bars=axs[0].bar(x+(j-.5)*width,values,width=width,label=label,color=color,edgecolor='black',linewidth=.3)
            for bar,v in zip(bars,values):axs[0].annotate(f'{v:.2f}',(bar.get_x()+bar.get_width()/2,v),xytext=(0,3),textcoords='offset points',ha='center',fontsize=7)
            rows.extend({'panel':'a','method':m,'measurement':field,'value':v,'unit':'percent','assessed_clips':415,'scope':'external model comparison, common IDs'} for m,v in zip(METHODS,values))
        axs[0].set_xticks(x,METHODS);axs[0].set_ylim(0,115);axs[0].set_ylabel('%');axs[0].set_title('(a) Detection on the same 415 clips');axs[0].legend(ncol=2,loc='upper center',frameon=False,fontsize=7)
        for metric in ('balanced_accuracy','f1'):
            b=ablation['bootstrap']['paired_differences']['uniform_weights'][metric]
            gain=(metrics['full'][metric]-metrics['uniform_weights'][metric])*100
            low,high=-b['ci95'][1]*100,-b['ci95'][0]*100
            assert abs(gain+b['observed_difference']*100)<1e-9
            deltas[metric]={'gain_pp':gain,'low95_pp':low,'high95_pp':high}
        for i,(metric,label) in enumerate([('balanced_accuracy','Balanced accuracy'),('f1','F1 score')]):
            d=deltas[metric]
            axs[1].errorbar(d['gain_pp'],i,xerr=[[d['gain_pp']-d['low95_pp']],[d['high95_pp']-d['gain_pp']]],fmt='o',capsize=4,color=COLORS[0],linewidth=1.4)
            axs[1].annotate(f"+{d['gain_pp']:.2f} pp",(d['gain_pp'],i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
            rows.append({'panel':'b','method':'Ours','measurement':metric+'_full_minus_uniform','value':d['gain_pp'],'unit':'percentage points','low95':d['low95_pp'],'high95':d['high95_pp'],'assessed_clips':980,'scope':'matched internal learned-weight versus uniform-weight reruns; seed 7'})
        axs[1].axvline(0,color='gray',linestyle='--',linewidth=.8);axs[1].set_yticks([0,1],['Balanced accuracy','F1 score']);axs[1].set_ylim(1.6,-.6);axs[1].set_xlim(-.5,6.2);axs[1].set_xlabel('Full minus uniform weights (percentage points)');axs[1].set_title('(b) Learned scoring weights: 980 clips');axs[1].grid(axis='x',alpha=.25);axs[1].spines[['top','right']].set_visible(False)
        axes_grid(axs[0])
        self.figure('scoring-effectiveness','Scoring: external results and a matched internal ablation',
            '(a) Balanced accuracy and F1 use the same 415 shared assessed clips, with frozen primary thresholds. Model differences do not isolate the scoring rule. (b) Matched Full minus uniform-weight scoring, 980 clips, five-epoch budget and seed 7: paired source-family bootstrap intervals (2,000 resamples) measure evaluation-population uncertainty, not training-seed variability. These populations differ and are not pooled. Learned weighting improves this matched comparison; the full scoring design is not best on every metric against every aggregation rule. All retained aggregation controls are provided in the companion CSV.',
            fig,rows,group,{'external_comparison':'BA and F1 on common assessed IDs','internal_comparison':'matched learned versus uniform scoring weights with evaluation bootstrap'})
        export_csv(OUT/'data/all-retained-aggregation-controls.csv',[{**r,'Method':'Ours: '+r['Method']} for r in agg])
        self.audit['scoring']={'matched_gains':deltas,'causal_external_scoring_superiority_proven':False,'other_controls_may_outperform_full':True}

    def extraction(self):
        group='extraction';cost=[r for r in self.csv(R/'table-iii-cost-001/table-iii-comparison.csv',group) if r['population']=='common_assessed'];by={method(r['model']):r for r in cost}
        coverage=self.csv(TRAIN/'publication/tables/table09-coverage.csv',group)
        self.source(R/'table-iii-cost-001/fk/artifacts-00-1/record.json',group)
        forensic=self.csv(OLD/'tables/table21-forensic-checks.csv',group);assert len(forensic)==6
        controls=self.csv(OLD/'data/fig14-shared-controls.csv',group)
        fig,axs=plt.subplots(1,3,figsize=(11.8,3.8),gridspec_kw={'width_ratios':[.85,1.05,1.4]});rows=[]
        values=[float(by[m]['feature_extraction']) for m in METHODS];bars=axs[0].bar(METHODS,values,color=COLORS,edgecolor='black',linewidth=.35)
        for bar,v in zip(bars,values):axs[0].annotate(f'{v:.3f} s',(bar.get_x()+bar.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
        axs[0].set_ylim(0,2.35);axs[0].set_ylabel('Seconds per clip');axs[0].set_title('(a) Feature extraction: 5 clips')
        rows.extend({'panel':'a','method':m,'measurement':'feature_extraction_time','value':v,'unit':'seconds','assessed_clips':5} for m,v in zip(METHODS,values))
        from matplotlib.colors import ListedColormap
        outputs=['Temporal\nlocalization','Signed bin\ncontributions','Source-bound\nrecord'];matrix=np.array([[1,0,0]]*3)
        axs[1].imshow(matrix,cmap=ListedColormap(['#f0f0f0','#b7d8ee']),vmin=0,vmax=1,aspect='auto')
        axs[1].set_xticks(range(3),METHODS,rotation=25,ha='right');axs[1].set_yticks(range(3),outputs,fontsize=7);axs[1].set_title('(b) Native evidence outputs')
        for i,output in enumerate(outputs):
            for j,m in enumerate(METHODS):
                axs[1].text(j,i,'Yes' if matrix[i,j] else 'N/A',ha='center',va='center',fontsize=8)
                rows.append({'panel':'b','method':m,'measurement':output.replace('\n',' '),'value':'available' if matrix[i,j] else 'not provided','unit':'capability, not accuracy','scope':'evaluated implementations only'})
        mapping={'FK':'Ours','visual_only':'Video only','audio_only':'Audio only','concatenation':'Concatenation','late_fusion':'Late fusion'};order=list(mapping)
        for j,(metric,label,color) in enumerate([('balanced_accuracy','BA','#2378ad'),('f1','F1','#e5a754')]):
            values=[float(next(r['value'] for r in controls if r['model']==model and r['metric']==metric))*100 for model in order];positions=np.arange(5)+(j-.5)*.34
            bars=axs[2].bar(positions,values,width=.32,label=label,color=color,edgecolor='black',linewidth=.25)
            for bar,v in zip(bars,values):axs[2].annotate(f'{v:.1f}',(bar.get_x()+bar.get_width()/2,v),xytext=(0,3),textcoords='offset points',ha='center',fontsize=5.8)
            rows.extend({'panel':'c','method':'Ours','variant':mapping[model],'measurement':metric,'value':v,'unit':'percent','assessed_clips':980,'scope':'own shared-encoder controls'} for model,v in zip(order,values))
        axs[2].set_xticks(range(5),list(mapping.values()),rotation=35,ha='right');axs[2].set_ylim(0,115);axs[2].set_ylabel('%');axs[2].set_title('(c) Our shared-encoder controls: 980 clips');axs[2].legend(ncol=2,frameon=False,loc='upper center',fontsize=7)
        for ax in (axs[0],axs[2]):axes_grid(ax)
        self.figure('feature-evidence-extraction','Feature extraction cost, evidence outputs and modality controls',
            '(a) Feature-extraction self time on five identical common-assessed profiling clips; AVTENet is a sum of separate stages. Ours is faster than AVTENet here, but slightly slower than AVFakeNet. This is not feature-quality accuracy. (b) Native outputs in the evaluated implementations; N/A means not provided, not zero performance. Signed contributions are traceable scores, not proof of manipulation causality. (c) Our shared-encoder modality/fusion controls on 980 assessed clips. All retained controls are shown; late fusion slightly exceeds Ours. This internal comparison does not establish that our image features are intrinsically better than the baseline representations.',
            fig,rows,group,{'external_comparison':'feature-extraction timing and native evidence availability','internal_comparison':'all shared-encoder audio/video/fusion controls'})
        self.audit['extraction']={'seconds_per_clip':{m:float(by[m]['feature_extraction']) for m in METHODS},'fastest_against_all_baselines':False,'intrinsic_feature_quality_superiority_proven':False}

    def finish(self):
        sys.path.insert(0,str(W/'09_Runtime/review-cache/archive-v0/1PiHw63lG2QUPDDZ'));import pymupdf as fitz
        assert len(self.pack.items)==3
        doc=fitz.open();rendered=[]
        for i,item in enumerate(self.pack.items,1):
            page=doc.new_page(width=960,height=650);assert page.insert_textbox(fitz.Rect(35,25,925,85),item['label']+'. '+item['title'],fontname='tibo',fontsize=18,color=(.03,.24,.38))>=0
            pix=fitz.Pixmap(str(OUT/item['png']));scale=min(890/pix.width,390/pix.height);width,height=pix.width*scale,pix.height*scale;x=(960-width)/2
            page.insert_image(fitz.Rect(x,98,x+width,98+height),pixmap=pix)
            assert page.insert_textbox(fitz.Rect(35,115+height,925,610),item['caption'],fontname='tiro',fontsize=10.2,lineheight=1.2)>=0,item['stem']
            page.insert_text((35,634),'Own experimental data only | No new training or inference',fontname='tiro',fontsize=8)
        doc.save(OUT/'THREE_MECHANISMS.pdf',garbage=4,deflate=True);doc.close()
        gallery='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Three mechanisms</title><style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:17px Georgia;color:#182733;line-height:1.5}h1{font-size:29px}h2{font-size:22px}img{max-width:100%}article{border-top:1px solid #bbb;padding:20px 0}a{color:#006d9c}</style><h1>Training, scoring and extraction: three comparisons</h1><p><a href="THREE_MECHANISMS.pdf">PDF</a> · <a href="THREE_MECHANISMS_PNG.zip">PNG bundle</a> · <a href="FINDINGS_TH.md">Thai interpretation</a> · <a href="LIMITATIONS.md">Limitations</a></p>'''
        for item in self.pack.items:
            gallery+='<article><h2>'+html.escape(item['label']+'. '+item['title'])+'</h2><img src="'+item['png']+'"><p>'+html.escape(item['caption'])+'</p><p><a href="'+item['png']+'">PNG</a> · <a href="'+item['svg']+'">SVG</a> · <a href="'+item['data']+'">CSV</a></p></article>'
        (OUT/'index.html').write_text(gallery+'</html>',encoding='utf-8')
        train=self.audit['training'];score=self.audit['scoring']['matched_gains'];extract=self.audit['extraction']['seconds_per_clip']
        (OUT/'FINDINGS_TH.md').write_text(f'''# กราฟเปรียบเทียบ 3 กลไก

## M1 — Training: เลือกเทียบจำนวนพารามิเตอร์ที่ต้องอัปเดต

Ours อัปเดต {train['distinct_optimized_parameters']['Ours']:,} พารามิเตอร์ ส่วน encoder {train['frozen_ours_encoder_parameters']:,} พารามิเตอร์ถูก freeze ไว้ เทียบกับ AVFakeNet {train['distinct_optimized_parameters']['AVFakeNet']:,} และ AVTENet {train['distinct_optimized_parameters']['AVTENet']:,} พารามิเตอร์ที่อัปเดตรวมทุก stage โดยนับน้ำหนักแต่ละชุดครั้งเดียว

จุดที่รองรับคือภาระการอัปเดตน้ำหนักที่เล็กกว่า ไม่ใช่ข้อพิสูจน์ว่าเทรนทั้งชุดเร็วกว่า baseline การใช้ pretrained อย่างเดียวไม่ใช่เหตุผลเฉพาะของเรา เพราะ baseline บางตัวก็ใช้ pretrained เช่นกัน กลไกที่เห็นใน implementation คือ freeze encoder และใช้ feature cache

การวัด online/cached ของเราเองเร็วขึ้น {train['ours_cached_speedup_in_scope']:.2f} เท่า เฉพาะหนึ่งวิดีโอ train 8 หน้าต่าง ไม่รวม preprocessing และ cache I/O และไม่ใช่การวัดเทรน baseline ทั้งชุด AVTENet เฉพาะ fusion stage มีเพียง 2,305 trainable parameters ซึ่งเบากว่าหัวของเรา แต่ยังมี expert training stages

## M2 — Scoring: แยกผลเทียบโมเดลกับหลักฐานของกลไก

ด้านซ้ายเทียบ BA/F1 บน 415 คลิปเดียวกันกับ baseline ด้านขวาเป็น matched ablation ของเรา 980 คลิป: learned weights เทียบ uniform weights เพิ่ม BA {score['balanced_accuracy']['gain_pp']:.2f} จุดเปอร์เซ็นต์ และ F1 {score['f1']['gain_pp']:.2f} จุดเปอร์เซ็นต์ พร้อมช่วงความเชื่อมั่นจาก paired bootstrap

ผลเทียบ baseline รวมความต่างทั้งโมเดล จึงระบุไม่ได้ว่าความแม่นที่เพิ่มทั้งหมดมาจาก scoring ส่วน ablation สนับสนุน learned weighting เฉพาะเงื่อนไขที่ทดสอบ มี aggregation rule อื่นที่ได้บาง metric สูงกว่า Full จึงเก็บผลทุก variant ใน data/all-retained-aggregation-controls.csv

## M3 — Extraction: เวลา + evidence ที่ได้ + audio/video controls

เวลาขั้น extraction ต่อคลิป: Ours {extract['Ours']:.3f} s, AVFakeNet {extract['AVFakeNet']:.3f} s, AVTENet {extract['AVTENet']:.3f} s บน profiling clips ร่วมกัน 5 คลิป เราเร็วกว่า AVTENet แต่ช้ากว่า AVFakeNet เล็กน้อยในขั้นนี้

Ours มี temporal localization, signed bin contributions และ source-bound record ใน implementation ที่ประเมิน ขณะที่ baseline ไม่ให้ native outputs เหล่านี้ การมี output ไม่ใช่ข้อพิสูจน์ว่า localization แม่นสมบูรณ์หรือ feature ภาพดีกว่าโดยเนื้อแท้

panel สุดท้ายเปรียบเทียบวิธีของเราเองกับ audio-only, video-only, concatenation และ late fusion โดยแสดงครบทุกตัว รวม late fusion ที่ดีกว่า Ours เล็กน้อย

ยังไม่มี controlled experiment ที่แยกผลของ filter/normalization/pretraining ออกจากปัจจัยอื่น จึงไม่อ้างว่าองค์ประกอบใดองค์ประกอบหนึ่งทำให้ cost ลดลงเชิงเหตุและผล ไม่มีการเทรนหรือ inference เพิ่มเพื่อสร้างกราฟชุดนี้
''',encoding='utf-8')
        (OUT/'LIMITATIONS.md').write_text('''# Evidence boundaries

- Baseline labels are AVFakeNet and AVTENet. They denote our retained local adaptations, not author-reported or certified official reproductions. Architecture/runtime scope remains as in the earlier study.
- Training panel (a) sums distinct parameter sets optimized at some point in each complete staged protocol. It is a parameter inventory, not a simultaneous peak, optimizer-step time, epoch energy, GPU-memory measurement, or pretraining cost. It does not mean the total Ours model is smallest. The frozen Ours encoders are still resident, executed to build caches, and counted separately. AVTENet cached fusion is very small, while its experts were fine-tuned upstream.
- Online/cache replay uses one original TRAIN video, eight usable windows, one warmup and three measured updates, identical initial weights and optimizer settings. Input preparation and disk/cache I/O are excluded. Features and final weights match exactly. The full training population and multiple seeds were not benchmarked in an equal-work three-method protocol.
- External scoring metrics use the same 415 assessed IDs; overall availability differs. Primary thresholds stay frozen. Model-level differences do not isolate the scoring rule. The matched Full/uniform weighting ablation uses 980 assessed clips, a five-epoch budget and seed 7. Bootstrap intervals quantify test-population sampling, not training variability. Other retained aggregation rules improve some metrics; the full design is not universally optimal.
- Extraction time uses only five common-assessed profiling clips, model loading excluded, uncontrolled OS cache and staged AVTENet execution. The extraction boundaries include each implementation's neural feature path; they do not establish intrinsic image/audio representation quality. Ours is slower than AVFakeNet at this stage and faster than AVTENet in this sample.
- Native evidence availability describes evaluated implementations only. It is not an impossibility statement about a baseline method. Record integrity is not media authenticity; signed contributions are not causal explanations. Localization assessed-region F1 in the existing study is 0.469.
- Own modality controls use shared encoders and 980 assessed clips. All audio-only, video-only, concatenation and late-fusion controls are shown. Late fusion slightly exceeds Ours. Single-seed controls and differing selected-model budgets limit architectural claims.
- No controlled pretraining/filter/normalization ablation is available here. Cost benefits cannot be attributed uniquely to those factors. No new training, inference or reference-paper measurements were used to make these figures.
''',encoding='utf-8')
        write(OUT/'catalog.json',{'items':self.pack.items});write(OUT/'measurements.json',self.audit)
        write(OUT/'source-manifest.json',{'files':self.sources.files,'groups':{k:sorted(v) for k,v in self.sources.groups.items()},'origin':'own retained experiments only'})
        qa=OUT/'qa';qa.mkdir(exist_ok=True);doc=fitz.open(OUT/'THREE_MECHANISMS.pdf')
        for i,page in enumerate(doc):
            p=qa/f'mechanism-{i+1:02d}.png';page.get_pixmap(dpi=100).save(p);rendered.append(p.relative_to(OUT).as_posix())
            for b in page.get_text('blocks'):assert b[0]>=0 and b[1]>=0 and b[2]<=page.rect.width+.5 and b[3]<=page.rect.height+.5
        doc.close()
        assert all(sha(p)==h for p,h in self.integrity.items())
        with zipfile.ZipFile(OUT/'THREE_MECHANISMS_PNG.zip','w',zipfile.ZIP_DEFLATED) as z:
            for item in self.pack.items:z.write(OUT/item['png'],Path(item['png']).name)
        with zipfile.ZipFile(OUT/'THREE_MECHANISMS_PNG.zip') as z:assert z.testzip() is None
        for item in self.pack.items:
            svg=(OUT/item['svg']).read_text(encoding='utf-8');assert not any(word in svg for word in ('adaptation','FK','adapt.'))
        write(OUT/'validation.json',{'status':'PASSED','figures':3,'all_external_comparisons_include_ours':True,'source_files_unchanged':True,'source_files':len(self.sources.files),'new_training':False,'new_inference':False,'reference_paper_data_used':False,'manual_visual_review':'pending','qa_pages':rendered})
        print(json.dumps({'status':'GENERATED','figures':3,'training_inventory':train['distinct_optimized_parameters'],'extraction_seconds':extract,'scoring_gains':score}),flush=True)


if __name__=='__main__':
    study=Study();study.training();study.scoring();study.extraction();study.finish()
