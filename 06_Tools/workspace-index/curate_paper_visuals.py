"""A small paper set: 3 comparison figures, 3 tables, 2 own-method appendix figures.

Reuse retained experimental values/assets only. No training or inference.
"""
import collections,csv,hashlib,html,json,os,re,shutil,sys,zipfile
from pathlib import Path

W=next(p for p in Path(__file__).resolve().parents if (p/'workspace-layout.json').is_file())
sys.path.insert(0,str(W/'06_Tools/workspace'))
from workspace_paths import relocate
os.environ['MPLCONFIGDIR']=str(W/'09_Runtime/caches/main/matplotlib')
from journal_visuals_common import Pack,Sources,setup_style,latex_escape
import matplotlib.pyplot as plt
import numpy as np

RUNS=W/'04_Results/experiments/fk'
OLD=RUNS/'paper-style-visuals-001'
COST=RUNS/'table-iii-complete-001/publication'
OUT=W/'04_Results/publication/paper-compact-001'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def csv_rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


class Curator:
    def __init__(self):
        self.sources=Sources();self.pack=Pack(OUT,self.sources);self.appendix=[];self.selected=[]
        self.catalogs={str(p):read(p/'catalog.json')['items'] for p in (OLD,COST)}
        self.integrity={}
    def source(self,path,group):
        path=relocate(Path(path),__file__).resolve();self.integrity[str(path)]=sha(path)
        self.sources.register(path,group);return path
    def original(self,base,stem):return next(x for x in self.catalogs[str(base)] if x['stem']==stem)
    def copy_figure(self,base,stem,label,title,caption,reason,appendix=False,only_sources=None):
        old=self.original(base,stem);target=OUT/'appendix' if appendix else OUT
        group='own_method' if appendix else 'comparison'
        dest_stem=('fig-a'+str(len(self.appendix)+1) if appendix else 'fig%02d'%(1+sum(x['kind']=='figure' for x in self.pack.items)))+'-'+stem.split('-',1)[1]
        new_number=len(self.appendix)+1 if appendix else 1+sum(x['kind']=='figure' for x in self.pack.items)
        item={**old,'stem':dest_stem,'number':new_number,'origin_number':old['number'],'label':label,'title':title,'caption':caption,'section':'Our-method analysis' if appendix else 'Main comparisons',
              'analysis_type':'our_method_only' if appendix else 'baseline_comparison','selection_reason':reason,
              'origin_package':base.relative_to(W).as_posix(),'origin_stem':stem,'source_files':[]}
        for key,folder,suffix in [('png','figures','.png'),('svg','figures','.svg'),('data','data','.csv')]:
            src=self.source(base/old[key],group);dest=target/folder/(dest_stem+suffix);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
            item[key]=dest.relative_to(OUT).as_posix()
            assert sha(src)==sha(dest)
        metadata=base/'metadata'/(stem+'.json');self.source(metadata,group)
        for path in only_sources or old.get('source_files',[]):
            source=self.source(W/path,group);item['source_files'].append(source.relative_to(W).as_posix())
        self.selected.append((str(base),stem))
        if not appendix:
            rows=csv_rows(OUT/item['data']);methods={r.get('method',r.get('model','')) for r in rows}
            assert any(m.startswith('FK') for m in methods),(stem,methods)
            item['comparison_methods']=sorted(methods);self.pack.items.append(item)
        else:
            raw=(OUT/item['data']).read_text(encoding='utf-8-sig')
            assert 'AVFakeNet' not in raw and 'AVTENet' not in raw
            self.appendix.append(item)
        return item
    def table(self,*args,**kwargs):
        self.pack.table(*args,**kwargs);item=self.pack.items[-1];item['analysis_type']='baseline_comparison';return item

    def main(self):
        setup_style()
        common_path=self.source(OLD/'tables/table03-main-common.csv','detection')
        coverage_path=self.source(OLD/'tables/table02-main-own.csv','detection')
        common=csv_rows(common_path);availability={r['Method']:r for r in csv_rows(coverage_path)}
        table_note=('Accuracy, balanced accuracy (BA), F1 and AUC use the SAME 415 common assessed clips. '
                    'Availability is separately reported out of all 1,000 attempts. AVTENet uses its frozen 0.5 threshold. '
                    'AVFakeNet hard majority/OR decisions have no fused continuous score, so AUC is N/A. '
                    'These are our local adaptations, not author-paper results. Common availability is a selected subset.')
        self.table('detection','Detection comparison on the common assessed clips',
                   ['Method','Assessed / 1,000','Accuracy %','BA %','F1 %','AUC'],
                   [[('FK (ours)' if r['Method']=='FK' else r['Method']),availability[r['Method']]['Assessed / attempt'],r['Accuracy %'],r['BA %'],r['F1 %'],r['AUC']] for r in common],
                   table_note,['detection'],widths=[2,1.1,1,1,1,1])
        self.copy_figure(OLD,'fig03-confusion-common','Fig. 1','Confusion matrices: our model versus baselines',
                         self.original(OLD,'fig03-confusion-common')['caption']+' FK is our method; both baselines appear alongside it.','Show directly comparable errors without a second detection-metric bar chart.')
        self.copy_figure(OLD,'fig05-roc-pr','Fig. 2','ROC and precision-recall: our model versus baselines',
                         self.original(OLD,'fig05-roc-pr')['caption'],'Show score discrimination; keep AVFakeNet as its measured operating point only.')
        costs=self.source(RUNS/'table-iii-cost-001/table-iii-comparison.csv','cost')
        rows=[r for r in csv_rows(costs) if r['population']=='common_assessed'];assert len(rows)==3
        cost_note=('Same five common-assessed profiling clips; one warmup and three measured passes, model loading excluded. '
                   'This profiling subset differs from the 415-clip detection subset. RTF = seconds/source second. '
                   'Memory is the peak statistic retained in our profiling summary. AVTENet time is a STAGED SUM and memory a stage maximum, '
                   'not a simultaneous three-expert deployment. FK includes forensic record persistence; OS disk cache is uncontrolled.')
        self.table('runtime-memory','Measured runtime and memory on common profiling clips',
                   ['Method','Clips','Seconds / clip','RTF','RSS MiB','CUDA MiB'],
                   [[r['model'],r['assessed'],f"{float(r['seconds']):.3f}",f"{float(r['seconds_per_source_second']):.3f}",f"{float(r['sampled_peak_rss_bytes'])/2**20:.1f}",f"{float(r['cuda_peak_allocated_bytes'])/2**20:.1f}"] for r in rows],
                   cost_note,['cost'],widths=[2.7,.6,1.2,.7,1,1])
        self.copy_figure(COST,'fig01-inference-components','Fig. 3','Computational components: our model versus baselines',
                         self.original(COST,'fig01-inference-components')['caption']+' Baseline record costs of zero indicate unavailable native forensic output, not equivalent support.',
                         'One component breakdown supports Table III without separate plots for each expert.')
        self.analytical_table()
        self.copy_figure(OLD,'fig15-core-ablations','Fig. A1','Our-method analysis: FK component ablations',
                         'OUR METHOD ONLY. Compares matched FK variants with components removed against the matched Full rerun. '
                         +self.original(OLD,'fig15-core-ablations')['caption']+' The matched Full is an ablation rerun, separate from the primary selected FK model. This is not an external-baseline comparison.',
                         'Explain the contribution of our evidence components.',appendix=True)
        self.copy_figure(COST,'fig06-fk-online-cached','Fig. A2','Our-method analysis: online versus cached training',
                         'OUR METHOD ONLY. Compares two execution modes of the same FK model. '
                         +self.original(COST,'fig06-fk-online-cached')['caption']+' This single-video prepared-input microbenchmark does not establish a whole-epoch or end-to-end speedup.',
                         'Explain our feature-cache process and the training column of Table III.',appendix=True,
                         only_sources=['04_Results/experiments/fk/table-iii-complete-001/fk/training-cost.json'])
        self.finish()

    def analytical_table(self):
        # Same analysis as the retained table, typeset as equations rather than
        # long text paragraphs. No results are taken from the reference papers.
        original=self.original(COST,'table10-analytical-cost')
        self.source(COST/original['data'],'analytical');self.source(COST/'TABLE_III_analytical.tex','analytical')
        headers=['Scheme','Preprocessing','Feature extraction','Fusion / evidence','Localization /\naggregation','Forensic record','Training']
        rows=[
          ['AVFakeNet\nadaptation','Face detection / crops\n+ clip mel',r'$n_fT^{V}_{\rm DST}+T^{A}_{\rm DST}$',r'$O(1)$ hard OR',r'$O(n_f)$ majority'+'\nLocalization N/A','N/A','Separate branch updates'],
          ['AVTENet\nadaptation','Face / lip crops\n+ mel / filterbanks',r'$N_w(T_{\rm ViViT}+T_{\rm AST}$'+'\n'+r'$+T_{\rm AVH}+T_{\rm TCN})$',r'$O(N_wD_{\rm sum})$'+'\nLinear fusion',r'$O(N_w)$ window mean'+'\nLocalization N/A','N/A','S1: expert updates\nS2: frozen features\n+ fusion update'],
          ['Ours (FK)',r'$n_fT_{\rm det}+T_{\rm VAD}$'+'\n'+r'$+NT_q+T_{\rm decode}$',r'$N_u(T_{\rm EN}+T_{\rm GRU}$'+'\n'+r'$+T_{\rm w2v}+BT_{\rm proj})$',r'$N_u[O(Bd^2+B^2d)$'+'\n'+r'$+B(T_{\rm disc}+T_R)$'+'\n'+r'$+|\mathcal{M}|T_{\rm head}]$',r'$O(N_uB)$'+'\n+ implemented top-k','Actual source / derived /\nrecord hashes\n'+r'$+O(N+N_uB)$','Online: '+r'$T_{\rm enc}+T_{\rm update}$'+'\nCached (amortized):\n'+r'$T_{\rm enc}/n_{\rm ep}+T_{\rm update}$']]
        fig,ax=plt.subplots(figsize=(14.5,4.05));ax.axis('off')
        fig.text(.5,.975,'TABLE III',ha='center',va='top',fontsize=11,weight='bold',color='#006d9c')
        fig.text(.5,.91,'Analytical computational cost of our implemented schemes',ha='center',va='top',fontsize=12)
        widths=np.array([1.05,1.65,2.0,2.15,1.8,2.0,2.15]);widths/=widths.sum()
        tbl=ax.table(cellText=rows,colLabels=headers,colWidths=widths,cellLoc='center',bbox=[0,0,1,.99]);tbl.auto_set_font_size(False);tbl.set_fontsize(12)
        for (r,c),cell in tbl.get_celld().items():
            cell.set_facecolor('white');cell.set_edgecolor('black');cell.set_linewidth(.6);cell.PAD=.045
            cell.set_height([.16,.22,.27,.35][r])
            if r==0:cell.get_text().set_weight('bold')
        fig.subplots_adjust(left=.012,right=.988,top=.80,bottom=.02)
        stem='table03-analytical-cost'
        for suffix in ('png','svg'):fig.savefig(OUT/'tables'/(stem+'.'+suffix),dpi=600,bbox_inches='tight',pad_inches=.04)
        plt.close(fig)
        shutil.copy2(COST/original['data'],OUT/'tables'/(stem+'.csv'))
        shutil.copy2(COST/'TABLE_III_analytical.tex',OUT/'tables'/(stem+'.tex'))
        item={**original,'number':3,'label':'Table III','stem':stem,'section':'Main comparisons','analysis_type':'baseline_comparison',
              'title':'Analytical computational cost of our implemented schemes',
              'png':'tables/'+stem+'.png','svg':'tables/'+stem+'.svg','data':'tables/'+stem+'.csv','latex':'tables/'+stem+'.tex',
              'caption':original['caption']+' Symbols: N attempted windows, N_u usable windows, N_w expert windows, B temporal bins, d projected dimension. N/A denotes missing native capability. This is code analysis, not an empirical Big-O proof.',
              'selection_reason':'The requested seven-column Table III comparison, including our method.'}
        self.pack.items.append(item);self.selected.append((str(COST),'table10-analytical-cost'))

    def booklet(self,items,path,title,subtitle):
        import pymupdf as fitz
        doc=fitz.open();cover=doc.new_page(width=842,height=595)
        cover.insert_text((40,82),title,fontname='tibo',fontsize=25,color=(.03,.24,.38))
        cover.insert_textbox(fitz.Rect(40,110,800,173),subtitle,fontname='tiro',fontsize=13,lineheight=1.3)
        y=205
        for item in items:
            cover.insert_text((40,y),item['label']+'. '+item['title'],fontname='tiro',fontsize=12);y+=28
        note=('Our retained experimental data only. Reference papers provide visual style, not data. '
              'Baselines are our local adaptations. No new training or inference was performed for this selection.')
        cover.insert_textbox(fitz.Rect(40,445,800,550),note,fontname='tiro',fontsize=11,lineheight=1.3)
        for number,item in enumerate(items,2):
            page=doc.new_page(width=842,height=595)
            left=page.insert_textbox(fitz.Rect(36,25,806,78),item['label']+'. '+item['title'],fontname='tibo',fontsize=17,color=(.03,.24,.38));assert left>=0
            pix=fitz.Pixmap(str(OUT/item['png']));scale=min(770/pix.width,350/pix.height);width=pix.width*scale;height=pix.height*scale;x=(842-width)/2
            page.insert_image(fitz.Rect(x,88,x+width,88+height),pixmap=pix)
            caption=item['caption']
            remain=page.insert_textbox(fitz.Rect(36,104+height,806,557),caption,fontname='tiro',fontsize=10,lineheight=1.22)
            assert remain>=0,('Caption overflow',item['label'])
            page.insert_text((36,579),'Own-method analysis' if item['analysis_type']=='our_method_only' else 'Comparison includes FK (ours)',fontname='tiro',fontsize=8)
            page.insert_text((790,579),str(number),fontname='tiro',fontsize=8)
        doc.save(path,garbage=4,deflate=True);doc.close()

    def finish(self):
        sys.path.insert(0,str(W/'09_Runtime/review-cache/archive-v0/1PiHw63lG2QUPDDZ'))
        import pymupdf as fitz
        main=self.pack.items;all_items=main+self.appendix
        assert collections.Counter(x['kind'] for x in main)=={'figure':3,'table':3}
        assert len(self.appendix)==2 and all(x['analysis_type']=='our_method_only' for x in self.appendix)
        for item in all_items:
            group='own_method' if item['analysis_type']=='our_method_only' else ('analytical' if item['number']==3 and item['kind']=='table' else 'detection' if item['kind']=='table' and item['number']==1 else 'cost' if item['kind']=='table' else 'comparison')
            if not item.get('source_files'):item['source_files']=sorted(self.sources.groups[group])
            write(OUT/'metadata'/(item['stem']+'.json'),item)
        self.booklet(main,OUT/'MAIN_COMPARISONS.pdf','Compact comparison figures and tables','3 comparison figures + 3 main tables. Every comparison includes our FK model.')
        self.booklet(self.appendix,OUT/'OUR_METHOD_APPENDIX.pdf','Our-method analysis','2 optional figures about our own process. These are not external-baseline comparisons.')
        def card(item):
            links=['<a href="'+html.escape(item[k])+'">'+name+'</a>' for k,name in [('svg','SVG'),('data','CSV')] if item.get(k)]
            if item.get('latex'):links.append('<a href="'+item['latex']+'">LaTeX</a>')
            return '<article><h3>'+html.escape(item['label']+'. '+item['title'])+'</h3><img loading="lazy" src="'+item['png']+'"><p>'+html.escape(item['caption'])+'</p><p>'+' · '.join(links)+'</p></article>'
        gallery='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Compact comparison set</title>
<style>body{max-width:1150px;margin:30px auto;padding:0 20px;font:17px Georgia;color:#182733;line-height:1.5}h1{font-size:30px}h2{font-size:24px}h3{font-size:21px}article{border-top:1px solid #ccc;padding:18px 0}img{max-width:100%;height:auto}summary{cursor:pointer;font-weight:bold}a{color:#006d9c}.note{background:#edf4f8;padding:14px}</style>
<h1>Compact comparison set</h1><p>3 comparison figures · 3 main tables</p>
<p><a href="MAIN_COMPARISONS.pdf">Open the main booklet</a> · <a href="OUR_METHOD_APPENDIX.pdf">Own-method appendix</a></p>
<p class="note">All comparison figures include our FK model. No baseline-only diagnostic plots are selected. All numerical results come from our retained experiments.</p><h2>Main comparisons</h2>'''
        gallery+=''.join(card(x) for x in main)
        gallery+='<details><summary>Optional: 2 own-method analyses (A1 and A2)</summary><p>A1 compares FK component variants. A2 compares online and cached execution of the same FK model. Neither is a baseline-only graph.</p>'+''.join(card(x) for x in self.appendix)+'</details>'
        gallery+='<details><summary>Selection and provenance</summary><p><a href="README.md">Selection guide</a> · <a href="excluded-items.csv">Items left out</a> · <a href="source-manifest.json">Own-data provenance</a></p></details></html>'
        (OUT/'index.html').write_text(gallery,encoding='utf-8')
        excluded=[]
        for folder,catalog in self.catalogs.items():
            for item in catalog:
                if (folder,item['stem']) in self.selected:continue
                baseline_only=any(token in item['stem'] for token in ('avtenet-calibration','avtenet-calibrated','avfakenet-branches'))
                excluded.append({'package':Path(folder).relative_to(W).as_posix(),'item':item['label'],'title':item['title'],
                                 'reason':'Standalone baseline analysis; excluded' if baseline_only else 'Additional/duplicated detail outside the approved compact set'})
        with (OUT/'excluded-items.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(excluded[0]));writer.writeheader();writer.writerows(excluded)
        guide='''# Compact figure and table selection

## Main set: 3 comparison figures and 3 tables

- Table I: detection on the same 415 clips; assessment counts separately shown out of 1,000 attempts.
- Figure 1: FK and both baseline confusion matrices on the shared 415 clips.
- Figure 2: FK and baseline ROC/PR comparison; AVFakeNet has an operating point, not an invented score curve.
- Table II: observed runtime and memory on five common profiling clips.
- Figure 3: FK and both baseline computational component costs.
- Table III: analytical cost comparison across the seven requested pipeline categories.

## Optional appendix: our own method, clearly labeled

- A1 — FK component ablations: matched Full versus versions with our components removed. This compares our own design choices, not external models.
- A2 — online versus cached-feature training: the same FK model under two execution modes on one eight-window TRAIN video. The observed speedup applies only to that measured scope, not an entire training pipeline.

No baseline-only plots are included. Extra baseline calibration/branch diagnostics, detailed microbenchmarks, duplicated performance plots, training traces, subgroup plots, and additional localization/forensic panels remain outside this compact selection. Original full collections are preserved as archives.

All figures expose their retained CSV and SVG. Tables include CSV and LaTeX. Reference papers are visual references only; no paper-reported values are imported. No training or inference was run to create this selection.
'''
        (OUT/'README.md').write_text(guide,encoding='utf-8')
        references=[]
        for name in ('Paper22.pdf','Paper16.pdf'):
            p=Path('C:/Users/teeku/Downloads')/name
            if p.exists():references.append({'name':name,'sha256':sha(p),'use':'visual style only; not a numerical source'})
        write(OUT/'source-manifest.json',{'files':self.sources.files,'groups':{k:sorted(v) for k,v in self.sources.groups.items()},'style_references':references})
        write(OUT/'catalog.json',{'main':main,'own_method_appendix':self.appendix})
        qa=OUT/'qa';qa.mkdir(exist_ok=True);rendered=[]
        for file in ('MAIN_COMPARISONS.pdf','OUR_METHOD_APPENDIX.pdf'):
            doc=fitz.open(OUT/file)
            for i,page in enumerate(doc):
                target=qa/(Path(file).stem.lower()+f'-{i+1:02d}.png');page.get_pixmap(matrix=fitz.Matrix(1,1)).save(target);rendered.append(target.relative_to(OUT).as_posix())
                for block in page.get_text('blocks'):assert block[0]>=0 and block[1]>=0 and block[2]<=page.rect.width+.5 and block[3]<=page.rect.height+.5
            doc.close()
        assert all(sha(p)==h for p,h in self.integrity.items())
        validation={'status':'PASSED','main_figures':3,'main_tables':3,'own_method_appendix_figures':2,
                    'baseline_only_selected':False,'all_selected_comparison_figures_contain_fk':True,'original_sources_unchanged':True,
                    'new_training':False,'new_inference':False,'reference_paper_numbers_used':False,'rendered_pages':rendered,'manual_visual_review':'pending'}
        write(OUT/'validation.json',validation)
        write(OUT/'complete.json',{'files':{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and 'qa' not in p.relative_to(OUT).parts and p.name!='complete.json'},'counts':{'comparison_figures':3,'tables':3,'own_method_figures':2}})
        print(json.dumps(validation),flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);Curator().main()
