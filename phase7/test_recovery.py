"""Validate both Kaggle ZIP and extracted-folder recovery with tiny checkpoints."""
from pathlib import Path
import shutil
import tempfile
import zipfile
import utils as u

def main():
    with tempfile.TemporaryDirectory(prefix='restore_',dir=Path(__file__).parent) as folder:
        folder=Path(folder);u.ROOT=folder/'run';u.ARTIFACT=folder/'CS781_Phase7_All_Artifacts.zip'
        u.write_json(u.ROOT/'run_config.json',{'example':1})
        (u.ROOT/'pcr_runner.py').write_text('# recovery test',encoding='utf8')
        u.save_torch(u.ROOT/'latest.pt',{'step':3})
        u.snapshot()
        def remove():
            target=u.ROOT.resolve()
            assert folder.resolve() in target.parents
            shutil.rmtree(target)
        mount=folder/'mount';mount.mkdir()
        shutil.copy2(u.ARTIFACT,mount/u.ARTIFACT.name);u.INPUT=mount
        remove();u.restore_artifact();assert u.read_json(u.ROOT/'run_config.json')['example']==1
        extracted=folder/'extracted';extracted.mkdir()
        with zipfile.ZipFile(u.ARTIFACT) as archive:archive.extractall(extracted)
        u.INPUT=extracted;remove();u.restore_artifact()
        assert u.read_json(u.ROOT/'run_config.json')['example']==1
        (u.ROOT/'.restore_in_progress').write_text('interrupted',encoding='utf8')
        u.restore_artifact();assert not (u.ROOT/'.restore_in_progress').exists()
        print('PASS ZIP, EXTRACTED-FOLDER AND INTERRUPTED RECOVERY',flush=True)

if __name__=='__main__':main()
