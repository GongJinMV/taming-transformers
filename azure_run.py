import os
import argparse
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import Workspace, Run, Dataset, Datastore, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.exceptions import UserErrorException


def get_settings_parser():
    """Parser for parameters get in commend-line format.

    This parser wraps all the setting parameters including credential information.
    This parser only works with experiment_entrance().

    Parameters
    ----------
    Void

    Returns
    -------
    ArgumentParser object
      a value in a string

    """
    settings_parser = argparse.ArgumentParser()
    settings_parser.add_argument("--config_path", default=r'C:\Work\GitHub\taming-transformers\config.json',
                                 type=str, help='your local path to config.json')
    settings_parser.add_argument("--download_path", default='', type=str,
                                 help='Leave it an an empty string if you don\'t want to save the output to local.')
    settings_parser.add_argument('--cluster_name', default='wsi-compute', type=str, help='Compute cluster name')
    settings_parser.add_argument('--environment_name', default='taming', type=str, help='Docker image name')
    settings_parser.add_argument('--experiment_name', default='taming-transformers-master', type=str,
                                 help='Experiment name display in AML')
    settings_parser.add_argument('--run_name', default='', type=str,
                                 help='customized run name. Leave it an empty string if you want to use random run name')
    settings_parser.add_argument('--register_model', default=False, type=bool,
                                 help='True to register the model in the cloud')
    settings_parser.add_argument('--blob_datastore_name', default='weldseamimages', type=str,
                                 help='Name of the datastore to workspace')
    settings_parser.add_argument('--container_name', default='daimler 4418 datasets', type=str,
                                 help='Name of Azure blob container')
    settings_parser.add_argument('--account_name', default='wsiblobstorage', type=str, help='Storage account name')
    settings_parser.add_argument('--account_key',
                                 default='fGlTZWdoSy9CLAYU1srPhEu2PCa/czuhNWhJ86VDpb86'
                                         '+4OcHMrUcwASIdPNmHgCRyNP4nlO7LAbJ28jTPql0g==',
                                 type=str, help='Storage account access key')
    settings_parser.add_argument("--return_counts", type=bool, default=True)
    settings_parser.add_argument("--mode", default='client')
    settings_parser.add_argument("--port", default=52162)
    settings_parser.add_argument("--host", default=52162)

    return (settings_parser)


def experiment_entrance(settings_args):
    """Preparation for starting a new run.

    This function connects to the workspace, environment, dataset and computing resources
        then returns a config.
    The args should at least contain config_path, blob_datastore_name, account_name,
        dataset_or_container_name, account_key, cluster_name, environment_name.

    Parameters
    ----------
    ParseArgs object

    Returns
    -------
    Workspace object

    ScriptRunConfig object

    """
    ws = Workspace.from_config(settings_args.config_path)
    #
    # try:
    #     blob_datastore = Datastore.get(ws, settings_args.blob_datastore_name)
    #     print("Found Blob Datastore with name: %s" % settings_args.blob_datastore_name)
    # except UserErrorException:
    #     blob_datastore = Datastore.register_azure_blob_container(
    #         workspace=ws,
    #         datastore_name=settings_args.blob_datastore_name,
    #         account_name=settings_args.account_name,
    #         container_name=settings_args.container_name,
    #         account_key=settings_args.account_key,
    #         grant_workspace_access=True,
    #         subscription_id='dbd94eae-ba51-4584-9dfa-f2cbefed36db',
    #         resource_group='WSI_ResourceGroup')
    #     print("Registered blob datastore with name: %s" % settings_args.blob_datastore_name)
    #
    # dataset = Dataset.File.from_files((blob_datastore, 'Anomaly Detection/Daimler/2022-09-12 16_07_33/'))

    try:
        compute_target = ComputeTarget(workspace=ws, name=settings_args.cluster_name)
        print('Found existing compute target')
    except ComputeTargetException:
        print('Creating a new compute target...')
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6',
                                                               max_nodes=4)
        compute_target = ComputeTarget.create(ws, settings_args.cluster_name, compute_config)
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

    myenv = Environment.get(workspace=ws, name=settings_args.environment_name)

    src = ScriptRunConfig(source_directory=os.path.abspath(os.path.join(os.getcwd(), ".")),
                          script='main.py',
                          compute_target=compute_target,
                          environment=myenv,
                          arguments=['--base', 'configs/custom_vqgan.yaml',
                                     '-t', True,
                                     '--gpus', '0,'])

    return (ws, src)


if __name__ == '__main__':
    settings_args = get_settings_parser().parse_args()
    ws, src = experiment_entrance(settings_args)
    # Experiment() must not be wrapped by any function,
    # otherwise it will be call twice and new two jobs.
    run = Experiment(workspace=ws, name=settings_args.experiment_name).submit(src)
    run.wait_for_completion(show_output=False)

    if settings_args.run_name.strip():
        run.display_name = settings_args.run_name.strip()

    if settings_args.download_path.strip():
        os.makedirs(settings_args.download_path, exist_ok=True)

    for f in run.get_file_names():
        output_dir = 'outputs/models/'
        if f.startswith(output_dir):
            if settings_args.download_path.strip():
                output_file_path = os.path.join(settings_args.download_path, f.split(output_dir)[-1])
                print('Downloading from {} to {} ...'.format(f, output_file_path))
                run.download_file(name=f, output_file_path=output_file_path)
            # It can only save model in h5 or HDF5 format.
            # If the model is in SavedModel(tensorflow) format, modify this part.
            if settings_args.register_model and (f.endswith('.h5') or f.endswith('.hdf5')):
                run.register_model(model_name=f.split('/')[-2].replace(' ', '_'), model_path=f)
