"""Selected comparison visuals with publication names and measured claims only.

Four comparison figures, three tables, two own-method figures. Raw experiments
remain immutable; numerical inputs are retained local data, never paper results.
"""
import collections,csv,hashlib,html,json,os,re,shutil,sys,zipfile
from pathlib import Path
W=next(p for p in Path(__file__).resolve().parents if (p/'workspace-layout.json').is_file())
os.environ['MPLCONFIGDIR']=str(W/'09_Runtime/caches/main/matplotlib')
sys.path.insert(0,str(W/'06_Tools/workspace'))
from workspace_paths import relocate
from journal_visuals_common import Pack,Sources,setup_style,axes_grid
import matplotlib.pyplot as plt
import numpy as np
RUNS=W/'04_Results/experiments/fk'
OLD=RUNS/'paper-style-visuals-001'
COST=RUNS/'table-iii-complete-001'
OUT=W/'04_Results/publication/paper-compact-002'
METHODS=['Ours','AVFakeNet','AVTENet']
COLORS={'Ours':'#2378ad','AVFakeNet':'#409b49','AVTENet':'#d85143'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def name(value):
    if value.startswith('FK') or value=='Ours (FK)':return 'Ours'
    if value.startswith('AVFakeNet'):return 'AVFakeNet'
    if value.startswith('AVTENet'):return 'AVTENet'
    return value
def csv_read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csv_write(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class Selection:
    def __init__(self):
        OUT.mkdir(parents=True,exist_ok=True);setup_style();self.sources=Sources();self.pack=Pack(OUT,self.sources);self.own=Pack(OUT/'appendix',self.sources);self.hashes={};self.main=[];self.appendix=[]
    def source(self,p,group):
        p=relocate(Path(p),__file__).resolve();self.hashes[str(p)]=sha(p);self.sources.register(p,group);return p
    def csv(self,p,group):return csv_read(self.source(p,group))
    def json(self,p,group):return read(self.source(p,group))
    def add_figure(self,slug,title,caption,fig,rows,group,own=False):
        pack=self.own if own else self.pack
        pack.figure(slug,title,caption,fig,rows,[group],section='Our-method analysis' if own else 'Comparison')
        item=pack.items[-1];item['analysis_type']='our_method_only' if own else 'baseline_comparison'
        item['source_files']=sorted(self.sources.groups[group])
        item['comparison_methods']=[] if own else METHODS
        if own:
            item['label']='Fig. A'+str(item['number'])
            for k in ('png','svg','data'):item[k]='appendix/'+item[k]
            self.appendix.append(item)
        else:self.main.append(item)
        return item
    def add_table(self,slug,title,headers,rows,caption,group,**kw):
        self.pack.table(slug,title,headers,rows,caption,[group],section='Comparison',**kw)
        item=self.pack.items[-1];item.update({'analysis_type':'baseline_comparison','source_files':sorted(self.sources.groups[group]),'comparison_methods':METHODS})
        self.main.append(item);return item

    def detection(self):
        common=self.csv(OLD/'tables/table03-main-common.csv','detection')
        own={r['Method']:r for r in self.csv(OLD/'tables/table02-main-own.csv','detection')}
        for p in ['04_Results/predictions/fk-heldout-copy.json','04_Results/experiments/fk/avfakenet-heldout-002/predictions.jsonl','04_Results/experiments/avtenet/auto-next-001/heldout/evaluate/predictions.jsonl']:
            self.source(W/p,'detection')
        caption=('Accuracy, balanced accuracy (BA), F1 and AUC use the same 415 common assessed clips. '
                 'Assessed counts are separately reported over all 1,000 attempts. The common subset is selected by availability. '
                 'AVTENet uses the frozen threshold 0.5. AVFakeNet hard decisions have no fused continuous score; AUC is N/A.')
        self.add_table('detection','Detection results on common assessed clips',['Method','Assessed / 1,000','Accuracy %','BA %','F1 %','AUC'],
            [[name(r['Method']),own[r['Method']]['Assessed / attempt'],r['Accuracy %'],r['BA %'],r['F1 %'],r['AUC']] for r in common],caption,'detection',widths=[1.35,1.4,1,1,1,1])
        confusion=[{**r,'method':name(r['method'])} for r in self.csv(OLD/'data/fig03-confusion-common.csv','detection')]
        fig,axes=plt.subplots(1,4,figsize=(8.2,3.0),gridspec_kw={'width_ratios':[1,1,1,.045]});axs=axes[:3]
        for ax,method in zip(axs,METHODS):
            selected=[r for r in confusion if r['method']==method];assert len(selected)==4
            fractions=np.zeros((2,2));counts=np.zeros((2,2),dtype=int)
            for row in selected:
                a,b=int(row['true_label']),int(row['prediction']);fractions[a,b]=float(row['row_fraction']);counts[a,b]=int(row['count'])
            assert counts.sum()==415
            image=ax.imshow(fractions,cmap='Blues',vmin=0,vmax=1,aspect='equal')
            for a in range(2):
                for b in range(2):ax.text(b,a,f'{counts[a,b]}\n{fractions[a,b]*100:.1f}%',ha='center',va='center',color='white' if fractions[a,b]>.6 else 'black',fontsize=9)
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','Fake'],yticklabels=['Real','Fake'],xlabel='Prediction',ylabel='True class',title=method)
        fig.colorbar(image,cax=axes[3],label='True-class fraction')
        self.add_figure('confusion-common','Confusion matrices on common assessed clips',
            'All methods use the same 415 assessed clips. Each cell gives the count and the percentage within its true class. The shared color scale ranges from 0 to 1. True class is on the vertical axis; prediction is on the horizontal axis. Labels are 0 = real and 1 = fake.',fig,confusion,'detection')
        rows=self.csv(OLD/'data/fig05-roc-pr.csv','detection');rows=[{**r,'method':name(r['method'])} for r in rows]
        fig,axs=plt.subplots(1,2,figsize=(7.5,3.1));auc={name(r['Method']):r['AUC'] for r in common}
        for ax,curve in zip(axs,('ROC','PR')):
            for method in METHODS:
                chosen=[r for r in rows if r['method']==method and r['curve']==curve]
                if chosen:
                    ax.plot([float(r['x']) for r in chosen],[float(r['y']) for r in chosen],label=method,color=COLORS[method],linewidth=1.4,
                            marker={'Ours':'o','AVTENet':'^'}[method],markevery=.12,markersize=4.5,markerfacecolor='white',markeredgewidth=.9)
                else:
                    chosen=[r for r in rows if r['method']==method and r['curve']=='hard '+curve+' point'];assert len(chosen)==1
                    ax.scatter([float(chosen[0]['x'])],[float(chosen[0]['y'])],marker='s',s=32,color=COLORS[method],label=method+' (operating point)',zorder=4)
            ax.set_xlim(-.02,1.02);ax.set_ylim(-.02,1.04);ax.grid(alpha=.25);ax.legend(fontsize=6.8,loc='lower right' if curve=='ROC' else 'lower left',frameon=False)
        axs[0].plot([0,1],[0,1],':',color='gray',linewidth=.7);axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='(a) ROC')
        axs[1].set(xlabel='Recall',ylabel='Precision',title='(b) Precision-recall')
        self.add_figure('score-comparison','Score discrimination across methods',
            'Retained scores on the same 415 common assessed clips. Ours and AVTENet have continuous scores; AVFakeNet is shown only at its measured hard-decision operating point. No artificial score curve is constructed. ROC-AUC summarizes discrimination across thresholds; a higher AUC alone does not establish calibration or every-threshold superiority.',
            fig,rows,'detection')

    def costs(self):
        rows=[r for r in self.csv(RUNS/'table-iii-cost-001/table-iii-comparison.csv','cost') if r['population']=='common_assessed']
        rows=[{**r,'model':name(r['model'])} for r in rows];by={r['model']:r for r in rows};assert set(by)==set(METHODS)
        caption=('The same five common-assessed profiling clips; one warmup and three measured passes, model loading excluded. '
                 'RTF is seconds per source second. AVTENet time sums separate stages, with memory summarized from stage peaks. '
                 'This is not a simultaneous ensemble. Ours includes persisted forensic records. OS disk cache is uncontrolled; these results are not a full-dataset speed guarantee.')
        self.add_table('runtime-memory','Measured runtime and host memory',['Method','Clips','Seconds / clip','RTF','RSS MiB'],
            [[r['model'],r['assessed'],f"{float(r['seconds']):.3f}",f"{float(r['seconds_per_source_second']):.3f}",f"{float(r['sampled_peak_rss_bytes'])/2**20:.1f}"] for r in rows],caption,'cost',widths=[1.5,.6,1.2,.7,1])
        components=['decode_io','preprocessing','feature_extraction','fusion_evidence','aggregation','forensic_record']
        labels=['Decode / I/O','Preprocessing','Feature extraction','Fusion / evidence','Aggregation','Forensic record']
        stage_colors=['#bdbdbd','#8dd3c7','#80b1d3','#7851b8','#d97818','#fb8072']
        fig,ax=plt.subplots(figsize=(9.3,4.65));x=np.array([0.,1.7,3.4]);bottom=np.zeros(3);data=[];positions={};width=.56
        for key,label,color in zip(components,labels,stage_colors):
            values=np.array([float(by[m][key]) for m in METHODS])
            # Exact seconds determine every stacked segment height.
            ax.bar(x,values,bottom=bottom,width=width,color=color,label=label,edgecolor='#555555' if key not in ('fusion_evidence','aggregation') else 'none',linewidth=.35)
            for i,method in enumerate(METHODS):
                seconds=values[i];status='measured'
                if key=='fusion_evidence' and method=='AVFakeNet':status='grouped with aggregation'
                elif key=='forensic_record' and method!='Ours':status='not provided'
                assert seconds>0 or status!='measured'
                visible_tick=status=='measured' and key in ('fusion_evidence','aggregation')
                positions[method,key]=(x[i],bottom[i]+seconds/2)
                data.append({'method':method,'measurement':key,'seconds':seconds,
                             'plotted_seconds':seconds if status=='measured' else '',
                             'stack_bottom_seconds':bottom[i],'stack_top_seconds':bottom[i]+seconds,
                             'visibility_marker':'colored location tick' if visible_tick else '',
                             'status':status,'clips':5,'scope':'median exclusive self time'})
                if visible_tick:
                    # Separate left/right strokes expose adjacent tiny components.
                    # Stroke thickness is a location aid, never an invented duration.
                    start=x[i]-width/2-.055 if key=='fusion_evidence' else x[i]+.02
                    stop=x[i]-.02 if key=='fusion_evidence' else x[i]+width/2+.055
                    ax.hlines(bottom[i]+seconds/2,start,stop,color=color,linewidth=2.4,zorder=5)
            bottom+=values
        callouts=[
            ('Ours','fusion_evidence',(-.53,7.0),'Fusion / evidence\n25.627 ms','center'),
            ('Ours','aggregation',(.68,2.4),'Aggregation\n1.822 ms','center'),
            ('AVFakeNet','aggregation',(2.28,8.05),'Aggregation\n27.40 us','center'),
            ('AVTENet','fusion_evidence',(2.84,15.45),'Fusion / evidence\n4.141 ms','center'),
            ('AVTENet','aggregation',(4.08,12.0),'Aggregation\n3.70 us','left')]
        colors=dict(zip(components,stage_colors))
        for method,key,target,text,align in callouts:
            center,height=positions[method,key];anchor=center-width/2-.025 if key=='fusion_evidence' else center+width/2+.025
            ax.annotate(text,xy=(anchor,height),xytext=target,ha=align,va='center',fontsize=8,color=colors[key],
                        arrowprops={'arrowstyle':'-','color':colors[key],'linewidth':.85,'connectionstyle':'angle,angleA=0,angleB=90,rad=3'},
                        bbox={'facecolor':'white','edgecolor':'none','pad':1.6,'alpha':.94})
        from matplotlib.ticker import MultipleLocator
        ax.set_ylim(0,16.5);ax.set_xlim(-1.2,4.9)
        ax.yaxis.set_major_locator(MultipleLocator(2));ax.yaxis.set_minor_locator(MultipleLocator(.5))
        ax.tick_params(axis='y',which='minor',length=3,width=.55)
        ax.set_xticks(x,METHODS);ax.set_ylabel('Seconds per clip');ax.set_xlabel('Method')
        ax.grid(axis='y',which='major',alpha=.3);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.22),frameon=False,fontsize=8)
        self.add_figure('pipeline-cost','Table III pipeline overview',
            'Same five profiling clips. Stacked bars retain the exact component medians on one linear seconds axis; minor ticks mark 0.5 s. Colored location strokes and callouts expose very small fusion/evidence and aggregation segments: stroke thickness is not duration. Values use ms/us where helpful (1 s = 1,000 ms = 1,000,000 us). AVFakeNet fusion is grouped with aggregation; baseline records are not provided. Stacks of component medians need not equal the median total in Table II. AVTENet sums separate stages. Training is outside this inference overview.',
            fig,data,'cost')

    def capabilities(self):
        group='capabilities'
        coverage=self.csv(COST/'publication/tables/table09-coverage.csv',group)
        localization=self.json(RUNS/'remaining-offline-001/localization/summary.json',group)
        forensic=self.csv(OLD/'tables/table21-forensic-checks.csv',group)
        assert len(forensic)==6 and all(r['Original']=='Verified' for r in forensic)
        rejected=sum(v=='Rejected' for r in forensic for v in r.values());assert rejected==32
        # Explicitly evaluated implementation outputs, not universal properties
        # of a named research method, and not performance numbers for N/A.
        matrix=[('Clip-level decision',[1,1,1]),('Continuous fused score',[1,0,1]),('Source-time localization',[1,0,0]),('Signed bin contributions',[1,0,0]),('Source / artifact-bound record',[1,0,0])]
        proof=self.source(RUNS/'table-iii-cost-001/fk/artifacts-00-1/record.json',group)
        record=read(proof);assert record
        fig,ax=plt.subplots(figsize=(7.6,3.45));values=np.array([v for label,v in matrix])
        from matplotlib.colors import ListedColormap
        ax.imshow(values,cmap=ListedColormap(['#f0f0f0','#b7d8ee']),vmin=0,vmax=1,aspect='auto')
        ax.set_xticks(range(3),METHODS);ax.xaxis.tick_top();ax.set_yticks(range(len(matrix)),[label for label,v in matrix]);ax.tick_params(length=0,pad=9)
        data=[]
        for i,(label,vals) in enumerate(matrix):
            for j,method in enumerate(METHODS):
                ax.text(j,i,'Available' if vals[j] else 'Not provided',ha='center',va='center',fontsize=8.5,color='#164b6e' if vals[j] else '#666666')
                data.append({'method':method,'output':label,'availability':'available' if vals[j] else 'not provided','scope':'evaluated implementation; not an accuracy score'})
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,len(matrix),1),minor=True);ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',length=0)
        self.add_figure('output-capabilities','Output capabilities in the evaluated implementations',
            'Availability of native outputs in our evaluated implementations, not an accuracy ranking or a claim about every implementation of these methods. Ours supplies source-time localization, signed bin contributions and source/artifact-bound records. Six original records verified; all 32 applicable mutations were rejected. This validates record integrity, not authenticity. Localization assessed-region F1 is 0.469, so output availability does not imply perfect localization.',
            fig,data,group)

    def analytical(self):
        group='analytical';self.source(COST/'publication/tables/table10-analytical-cost.csv',group)
        rows=[
            ['AVFakeNet','Face detection / crops\n+ clip mel',r'$n_fT^V_{\rm DST}+T^A_{\rm DST}$',r'$O(1)$ hard OR',r'$O(n_f)$ majority'+'\nLocalization N/A','N/A','Separate branch updates'],
            ['AVTENet','Face / lip crops\n+ mel / filterbanks',r'$N_w(T_{\rm ViViT}+T_{\rm AST}$'+'\n'+r'$+T_{\rm AVH}+T_{\rm TCN})$',r'$O(N_wD_{\rm sum})$'+'\nLinear fusion',r'$O(N_w)$ window mean'+'\nLocalization N/A','N/A','S1: expert updates\nS2: frozen features\n+ fusion update'],
            ['Ours',r'$n_fT_{\rm det}+T_{\rm VAD}$'+'\n'+r'$+NT_q+T_{\rm decode}$',r'$N_u(T_{\rm EN}+T_{\rm GRU}$'+'\n'+r'$+T_{\rm w2v}+BT_{\rm proj})$',r'$N_u[O(Bd^2+B^2d)$'+'\n'+r'$+B(T_{\rm disc}+T_R)$'+'\n'+r'$+|\mathcal{M}|T_{\rm head}]$',r'$O(N_uB)$'+'\n+ implemented top-k','Actual source / derived /\nrecord hashes\n'+r'$+O(N+N_uB)$','Online: '+r'$T_{\rm enc}+T_{\rm update}$'+'\nCached (amortized):\n'+r'$T_{\rm enc}/n_{\rm ep}+T_{\rm update}$']]
        headers=['Scheme','Preprocessing','Feature extraction','Fusion / evidence','Localization /\naggregation','Forensic record','Training']
        fig,ax=plt.subplots(figsize=(14.5,4.05));ax.axis('off');fig.text(.5,.975,'TABLE III',ha='center',va='top',fontsize=11,weight='bold',color='#006d9c');fig.text(.5,.91,'Analytical computational cost of the evaluated schemes',ha='center',va='top',fontsize=12)
        widths=np.array([.9,1.65,2,2.15,1.8,2,2.15]);widths/=widths.sum();tbl=ax.table(cellText=rows,colLabels=headers,colWidths=widths,cellLoc='center',bbox=[0,0,1,.99]);tbl.auto_set_font_size(False);tbl.set_fontsize(12)
        for (r,c),cell in tbl.get_celld().items():
            cell.set_facecolor('white');cell.set_edgecolor('black');cell.set_linewidth(.6);cell.PAD=.04;cell.set_height([.16,.22,.27,.35][r])
            if r==0:cell.get_text().set_weight('bold')
        fig.subplots_adjust(left=.012,right=.988,top=.80,bottom=.02);stem='table03-analytical-cost'
        for ext in ('png','svg'):fig.savefig(OUT/'tables'/(stem+'.'+ext),dpi=600,bbox_inches='tight',pad_inches=.04)
        plt.close(fig)
        csv_write(OUT/'tables'/(stem+'.csv'),[dict(zip([h.replace('\n',' ') for h in headers],r)) for r in rows])
        latex=(COST/'publication/TABLE_III_analytical.tex').read_text(encoding='utf-8').replace('AVFakeNet adapt.','AVFakeNet').replace('AVTENet adapt.','AVTENet').replace('Ours (FK)','Ours')
        (OUT/'tables'/(stem+'.tex')).write_text(latex,encoding='utf-8')
        self.main.append({'kind':'table','number':3,'label':'Table III','stem':stem,'title':'Analytical computational cost of the evaluated schemes','section':'Comparison','analysis_type':'baseline_comparison','comparison_methods':METHODS,
            'png':'tables/'+stem+'.png','svg':'tables/'+stem+'.svg','data':'tables/'+stem+'.csv','latex':'tables/'+stem+'.tex',
            'source_files':sorted(self.sources.groups[group]),'source_groups':[group],
            'caption':'Analytical costs of the evaluated implementations, not measured seconds or an empirical Big-O proof. N is attempted windows, N_u usable windows, N_w expert windows, B bins and d projected dimension. T_update includes forward/loss, backward, clipping and optimizer; it is not exactly three forward passes. Cache I/O is separate. Source hashing is repeated during preparation; a fixed two-hash expression understates this implementation. N/A means the native output is not provided.'})

    def own_method(self):
        group='own_modalities';rows=self.csv(OLD/'data/fig14-shared-controls.csv',group);self.source(OLD/'tables/table11-shared-controls.csv',group)
        mapping={'FK':'Ours','visual_only':'Video only','audio_only':'Audio only','concatenation':'Concatenation','late_fusion':'Late fusion'}
        rows=[{**r,'model':mapping[r['model']]} for r in rows];order=list(mapping.values());fig,axs=plt.subplots(1,3,figsize=(10,3.05))
        for ax,metric,title in zip(axs,['balanced_accuracy','f1','roc_auc'],['Balanced accuracy','F1 score','ROC-AUC']):
            by={r['model']:float(r['value']) for r in rows if r['metric']==metric};values=[by[m]*100 for m in order];bars=ax.bar(range(5),values,color=['#2378ad','#96b6ca','#b7c9d5','#b3a2c7','#95b890'],edgecolor='black',linewidth=.3)
            for bar,v in zip(bars,values):ax.annotate(f'{v:.1f}',(bar.get_x()+bar.get_width()/2,v),ha='center',xytext=(0,3),textcoords='offset points',fontsize=6)
            ax.set_xticks(range(5),order,rotation=35,ha='right');ax.set_ylim(0,105);ax.set_title(title);ax.set_ylabel('%');axes_grid(ax)
        self.add_figure('modality-controls','Our-method analysis: modality and fusion controls',
            'OUR METHOD ONLY. Shared frozen encoders and 980 jointly assessed clips; controls use the retained five-epoch, seed-7 protocol. All retained controls are shown. Ours improves on audio-only, video-only and concatenation, while the late-fusion control is slightly higher on BA, F1 and AUC. These comparisons do not establish that our evidence architecture is the best fusion rule; budgets and selection details are in the limitations.',fig,rows,group,own=True)
        group='own_training';source=self.json(COST/'fk/training-cost.json',group);matched=source['matched_online_cached'];rows=[{'mode':m,'median_seconds':matched[m]['medians']['total_seconds'],'usable_windows':matched['usable_windows']} for m in ('online','cached')]
        assert matched['final_weight_max_absolute_error']==0 and max(matched['feature_max_absolute_errors'].values())==0
        fig,ax=plt.subplots(figsize=(5.7,3.15));values=[r['median_seconds']*1000 for r in rows];bars=ax.bar(['Online encoders','Cached features'],values,color=['#df927f','#2378ad'],edgecolor='black',linewidth=.4)
        for bar,v in zip(bars,values):ax.annotate(f'{v:.2f} ms',(bar.get_x()+bar.get_width()/2,v),ha='center',xytext=(0,4),textcoords='offset points')
        ax.set_ylabel('Milliseconds per complete-video update');ax.set_ylim(0,max(values)*1.17);axes_grid(ax)
        speedup=values[0]/values[1]
        self.add_figure('training-cache','Our-method analysis: online versus cached training',
            f'OUR METHOD ONLY. Same model, initial weights, optimizer and one TRAIN video with eight usable windows. Online {values[0]:.2f} ms; cached {values[1]:.2f} ms ({speedup:.2f}x in this scope). Features and final weights match exactly. Both exclude raw-media preprocessing and disk/cache I/O. This is not a baseline training-speed comparison or a whole-epoch speedup.',fig,rows,group,own=True)

    def pdf(self,items,path,title,subtitle):
        import pymupdf as fitz
        doc=fitz.open();page=doc.new_page(width=842,height=595);page.insert_text((38,78),title,fontname='tibo',fontsize=24,color=(.03,.24,.38));assert page.insert_textbox(fitz.Rect(38,110,804,170),subtitle,fontname='tiro',fontsize=12)>=0
        y=204
        for item in items:page.insert_text((38,y),item['label']+'. '+item['title'],fontname='tiro',fontsize=11.5);y+=30
        page.insert_textbox(fitz.Rect(38,460,804,550),'Numerical inputs: our retained experiments only. Baseline implementation scope and measurement caveats are documented in LIMITATIONS.md. No new training or inference was run.',fontname='tiro',fontsize=11)
        for i,item in enumerate(items,2):
            page=doc.new_page(width=842,height=595);assert page.insert_textbox(fitz.Rect(36,25,806,80),item['label']+'. '+item['title'],fontname='tibo',fontsize=16.5,color=(.03,.24,.38))>=0
            pix=fitz.Pixmap(str(OUT/item['png']));scale=min(770/pix.width,375/pix.height);width,height=pix.width*scale,pix.height*scale;x=(842-width)/2
            page.insert_image(fitz.Rect(x,90,x+width,90+height),pixmap=pix)
            assert page.insert_textbox(fitz.Rect(36,108+height,806,557),item['caption'],fontname='tiro',fontsize=10,lineheight=1.22)>=0,item['title']
            page.insert_text((36,579),'Our-method comparison' if item['analysis_type']=='our_method_only' else 'Ours compared with AVFakeNet and AVTENet',fontname='tiro',fontsize=8);page.insert_text((790,579),str(i),fontname='tiro',fontsize=8)
        doc.save(path,garbage=4,deflate=True);doc.close()

    def finish(self):
        sys.path.insert(0,str(W/'09_Runtime/review-cache/archive-v0/1PiHw63lG2QUPDDZ'));import pymupdf as fitz
        assert collections.Counter(x['kind'] for x in self.main)=={'figure':4,'table':3};assert len(self.appendix)==2
        # Restore paper order: detection, scoring, runtime, pipeline, output,
        # analytical Table III. Labels already preserve their own type order.
        for item in self.main+self.appendix:write(OUT/'metadata'/(item['stem']+'.json'),item)
        self.pdf(self.main,OUT/'MAIN_COMPARISONS.pdf','Comparison figures and tables','4 comparison figures and 3 tables. Methods: Ours, AVFakeNet and AVTENet.')
        self.pdf(self.appendix,OUT/'OUR_METHOD_APPENDIX.pdf','Our-method comparisons','A1: audio, video and fusion controls. A2: online versus cached training.')
        def card(x):
            return '<article><h3>'+html.escape(x['label']+'. '+x['title'])+'</h3><img loading="lazy" src="'+x['png']+'"><p>'+html.escape(x['caption'])+'</p><p><a href="'+x['svg']+'">SVG</a> · <a href="'+x['data']+'">CSV</a>'+(' · <a href="'+x['latex']+'">LaTeX</a>' if x.get('latex') else '')+'</p></article>'
        gallery='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Comparison figures and tables</title><style>body{max-width:1150px;margin:30px auto;padding:0 20px;font:17px Georgia;color:#182733;line-height:1.5}h1{font-size:30px}h2{font-size:23px}h3{font-size:20px}article{border-top:1px solid #ccc;padding:20px 0}img{max-width:100%}summary{cursor:pointer;font-weight:bold}a{color:#006d9c}</style><h1>Comparison figures and tables</h1><p>3 comparison figures + 3 tables. Labels: Ours, AVFakeNet, AVTENet.</p><p><a href="MAIN_COMPARISONS.pdf">Main PDF</a> · <a href="OUR_METHOD_APPENDIX.pdf">Our-method appendix</a> · <a href="COMPARISON_PNG.zip">All selected PNGs</a></p>'''
        gallery=gallery.replace('3 comparison figures + 3 tables','4 comparison figures + 3 tables')
        gallery+='<h2>Main comparisons</h2>'+''.join(card(x) for x in self.main)+'<details><summary>Our-method comparisons: audio/video controls and training cache</summary>'+''.join(card(x) for x in self.appendix)+'</details><p><a href="CLAIMS.md">What the evidence supports</a> · <a href="LIMITATIONS.md">Limitations</a> · <a href="README.md">Selection guide</a></p></html>'
        (OUT/'index.html').write_text(gallery,encoding='utf-8')
        (OUT/'README.md').write_text('''# Selected paper visuals

Main figures compare Ours with AVFakeNet and AVTENet. No standalone baseline diagnostics are included.

1. Confusion matrices: restored comparison on the same 415 clips.
2. Score discrimination: ROC and precision-recall, all methods on the same 415 clips.
3. Table III pipeline overview: stacked bars on a linear seconds axis, with minor ticks every 0.5 s. Colored location strokes and labeled leaders identify tiny fusion/evidence and aggregation segments without inflating their true durations. Grouped or unavailable timings are not invented as standalone measurements.
4. Output capabilities: continuous scoring, localization, signed contributions and bound forensic records in the evaluated implementations.

Tables I–III cover detection, measured runtime/memory, and analytical computational cost.

Own-method appendix:

- A1: Ours, audio only, video only, concatenation and late fusion. Every retained control is shown, including late fusion that slightly exceeds Ours.
- A2: the same model trained with online versus cached features on one TRAIN video. This is an internal process comparison, not a baseline training benchmark.

Names in publication figures and tables are Ours, AVFakeNet and AVTENet. Scope of the local baseline implementations is explained only in LIMITATIONS.md. Original experimental identifiers are preserved in provenance paths. Numerical results are unchanged. Reference papers supply style only.
''',encoding='utf-8')
        (OUT/'CLAIMS.md').write_text('''# Supported claims and figure selection

- **Errors (Figure 1):** restored confusion matrices compare the same 415 assessed clips with the current method names.
- **Scoring (Figure 2):** ROC-AUC on 415 shared assessed clips is 0.9461 for Ours and 0.8982 for AVTENet. AVFakeNet has no fused continuous score, so only its hard operating point is plotted. This is not proof of better calibration.
- **Preprocessing/runtime (Figure 3, Table II):** Ours has lower measured preprocessing (1.345 vs 7.320 and 10.595 s/clip) and total (6.232 vs 9.609 and 14.069 s/clip) in five common profiling clips. Figure 3 uses proportional stacked bars on a linear seconds axis. Thin colored location strokes identify fusion/evidence and aggregation; their stroke thickness is a visibility aid, not a rescaled measurement. Interpret under the stated hardware and staged AVTENet boundary.
- **Traceable outputs (Figure 4, Table III):** evaluated Ours provides native temporal localization, signed contributions and source/artifact-bound records that the two evaluated baselines do not provide. This is output availability, not a universal impossibility result. Original records verify 6/6 and applicable mutations are rejected 32/32. Integrity is distinct from media authenticity. Localization F1 is 0.469 in assessed regions.
- **Modality controls (A1):** the multimodal model exceeds audio-only, video-only and concatenation in these measurements. Late fusion is slightly better than Ours. Do not claim the architecture outperforms every fusion control.
- **Training process (A2):** caching reduces the measured update from 383.47 to 17.54 ms on one eight-window training video. Both modes exclude media preparation and cache I/O. Do not claim faster full training than the baselines.

Do not claim the lowest GPU memory: Ours uses more allocated GPU memory than the baselines in the retained profiling summary. Do not claim perfect localization, complete coverage, or unbiased superiority across an entire dataset.
''',encoding='utf-8')
        (OUT/'LIMITATIONS.md').write_text('''# Limitations and implementation scope

Publication labels are Ours, AVFakeNet and AVTENet. The baselines are our local, paper-guided adaptations; these are not author-reported results or certified official-code reproductions. AVFakeNet uses our reconstructed/normalized DST branches and retained hard majority/OR decision. AVTENet uses the explicit joint-attention HF ViViT adaptation with AST and native AV-HuBERT plus its temporal head. These implementation choices and the limited training budgets constrain external claims.

Detection uses 415 shared assessed clips selected from 1,000 attempts. Availability differs (980, 417 and 923 assessed for Ours, AVFakeNet and AVTENet). A shared subset controls IDs but does not eliminate missing-data bias. The primary AVTENet threshold is 0.5; exploratory post-inspection calibration is excluded.

Timing uses five common-assessed clips from the retained profiling sample. Model loading is excluded; OS cache is uncontrolled. AVTENet runtime is a sum of separately executed stages and memory is summarized from stage peaks. Ours includes record persistence. No claim of simultaneous ensemble memory or population-wide runtime is made.

Output availability describes only the evaluated implementations. Native baseline localization and forensic records are not provided; these entries are not zero accuracy. Localized assessed-region F1 for Ours is 0.469, with time assessment coverage 89.7%. Hash verification validates record integrity, not whether the input media is genuine.

Own-method controls share encoders and an assessment population of 980, with retained five-epoch, seed-7 controls; Ours is the retained selected model, and training budgets/model selection are not a fresh fully matched multi-seed experiment. Late fusion slightly exceeds Ours in BA, F1 and AUC. Cached-versus-online training uses one eight-window TRAIN video with discarded updates and identical initialization; it excludes raw media preprocessing and disk/cache I/O. Full-epoch baseline training time was not measured under a common protocol.

Table III is analytical code-based cost accounting, not an empirical proof of asymptotic complexity. Original sources remain unchanged; only display labels and selection were revised. No new training, inference or validation-threshold fitting was performed.
''',encoding='utf-8')
        write(OUT/'catalog.json',{'main':self.main,'own_method_appendix':self.appendix})
        write(OUT/'source-manifest.json',{'files':self.sources.files,'groups':{k:sorted(v) for k,v in self.sources.groups.items()},'numeric_source':'own retained experiments only','display_names':METHODS})
        qa=OUT/'qa';qa.mkdir(exist_ok=True);png=OUT/'PNG_EXPORT';(png/'main').mkdir(parents=True,exist_ok=True);(png/'our-method').mkdir(exist_ok=True);(png/'pdf-pages').mkdir(exist_ok=True)
        pages=[];zip_files=[]
        for filename in ('MAIN_COMPARISONS.pdf','OUR_METHOD_APPENDIX.pdf'):
            doc=fitz.open(OUT/filename)
            for i,page in enumerate(doc):
                p=qa/(Path(filename).stem.lower()+f'-{i+1:02d}.png');page.get_pixmap(dpi=90).save(p);pages.append(p.relative_to(OUT).as_posix())
                exported=png/'pdf-pages'/(Path(filename).stem+f'-{i+1:02d}.png');page.get_pixmap(dpi=300).save(exported);zip_files.append(exported)
                for block in page.get_text('blocks'):assert block[0]>=0 and block[1]>=0 and block[2]<=page.rect.width+.5 and block[3]<=page.rect.height+.5
            doc.close()
        for item in self.main+self.appendix:
            target=png/('our-method' if item['analysis_type']=='our_method_only' else 'main')/Path(item['png']).name;shutil.copy2(OUT/item['png'],target);zip_files.append(target)
        for p,h in self.hashes.items():assert sha(p)==h,p
        with zipfile.ZipFile(OUT/'COMPARISON_PNG.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in zip_files:z.write(p,p.relative_to(png).as_posix())
        with zipfile.ZipFile(OUT/'COMPARISON_PNG.zip') as z:assert z.testzip() is None
        write(png/'manifest.json',{'files':{p.relative_to(png).as_posix():sha(p) for p in zip_files}})
        validation={'status':'PASSED','main_figures':4,'main_tables':3,'own_method_figures':2,'display_names':METHODS,'standalone_baseline_plots':False,'original_sources_unchanged':True,'new_training':False,'new_inference':False,'manual_visual_review':'pending','rendered_pages':pages}
        write(OUT/'validation.json',validation);self.check_labels();print(json.dumps(validation),flush=True)

    def check_labels(self):
        for item in self.main+self.appendix:
            text=' '.join(str(item[k]) for k in ('title','caption'))+(OUT/item['data']).read_text(encoding='utf-8-sig')
            svg=(OUT/item['svg']).read_text(encoding='utf-8')
            assert not re.search(r'\bFK\b|adaptation|\badapt\.',text+' '+svg,re.I),item['stem']
            if item['analysis_type']=='baseline_comparison':assert all(m in (OUT/item['data']).read_text(encoding='utf-8-sig') for m in METHODS)
            else:assert 'AVFakeNet' not in svg and 'AVTENet' not in svg


if __name__=='__main__':
    study=Selection();study.detection();study.costs();study.capabilities();study.analytical();study.own_method();study.finish()
