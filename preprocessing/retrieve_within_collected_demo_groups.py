import numpy as np
from collections import defaultdict
import json
from openpi.policies.utils import myprint, embed, load_dinov2, embed_with_batches, EMBED_DIM
import os
import argparse
from autofaiss import build_index
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # This prevents JAX from preallocating most of the GPU memory.
EMBED_TYPES = ["top_image", "wrist_image"]

def _load_episode_embeddings(ep_fol, embedding_type):
	if embedding_type in EMBED_TYPES:
		return np.load(f"{ep_fol}/processed_demo.npz")[f"{embedding_type}_embeddings"]
	elif embedding_type == "both":
		return np.concatenate([np.load(f"{ep_fol}/processed_demo.npz")[f"{item}_embeddings"] for item in EMBED_TYPES], axis=1)
	else:
		raise ValueError(f'{embedding_type=} is not in {EMBED_TYPES} and not "both"')

def pair_groups_by_suffix(ds_name, query_suffix, corpus_suffix):
	groups = [g for g in os.listdir(ds_name) if os.path.isdir(f"{ds_name}/{g}") and not (g.startswith('.git') or g.endswith('.json') or g == 'README.md')]
	by_base = {}
	for g in groups:
		if g.endswith(query_suffix):
			base = g[:-len(query_suffix)]
			by_base.setdefault(base, {})['query'] = f"{ds_name}/{g}"
		elif g.endswith(corpus_suffix):
			base = g[:-len(corpus_suffix)]
			by_base.setdefault(base, {})['corpus'] = f"{ds_name}/{g}"
	return [(base, v['query'], v['corpus']) for base, v in by_base.items() if 'query' in v and 'corpus' in v]

def retrieval_preprocessing_cross_domain(ds_name, mappings, nb_cores_autofaiss, knn_k, embedding_type, query_suffix, corpus_suffix, output_stub):
	myprint(f'[cross_domain] starting cross-domain retrieval preprocessing for {embedding_type} with query_suffix={query_suffix} corpus_suffix={corpus_suffix}')
	groups_to_ep_fols = mappings['groups_to_ep_fols']
	fols_to_ep_idxs = mappings['fols_to_ep_idxs']

	# pair groups by base name
	pairs = pair_groups_by_suffix(ds_name, query_suffix, corpus_suffix)
	myprint(f'[cross_domain] found {len(pairs)} group pairs')

	for pair_count, (base, query_group, corpus_group) in enumerate(pairs):
		myprint(f'[cross_domain] processing base={base} [pair {pair_count}/{len(pairs)}]')
		# build corpus (human) embeddings index once per base
		corpus_ep_fols = groups_to_ep_fols.get(corpus_group, [])
		if len(corpus_ep_fols) == 0:
			myprint(f'[cross_domain] skipping base={base} because corpus group has no episodes: {corpus_group}')
			continue
		corpus_embeddings_list = []
		corpus_indices_list = []
		corpus_embeddings_map = {}
		for ep_fol in corpus_ep_fols:
			try:
				ep_embeddings = _load_episode_embeddings(ep_fol, embedding_type)
			except Exception as e:
				myprint(f'[cross_domain] skipping episode {ep_fol} due to error loading embeddings: {e}')
				continue
			ep_idx = fols_to_ep_idxs[ep_fol]
			corpus_embeddings_list.append(ep_embeddings)
			corpus_embeddings_map[ep_idx] = ep_embeddings
			num_steps = len(ep_embeddings)
			corpus_indices_list.extend([[ep_idx, stp_idx] for stp_idx in range(num_steps)])
		if len(corpus_embeddings_list) == 0:
			myprint(f'[cross_domain] skipping base={base} because no corpus embeddings were loaded')
			continue
		corpus_embeddings = np.concatenate(corpus_embeddings_list, axis=0)
		corpus_indices = np.array(corpus_indices_list)
		embedding_dim = corpus_embeddings.shape[1]
		myprint(f'[cross_domain] built corpus with {len(corpus_embeddings)} embeddings, dim={embedding_dim}')
		knn_index, _ = build_index(embeddings=corpus_embeddings,
								save_on_disk=False,
								min_nearest_neighbors_to_retrieve=knn_k + 5,
								max_index_query_time_ms=10,
								max_index_memory_usage="25G",
								current_memory_available="50G",
								metric_type='l2',
								nb_cores=nb_cores_autofaiss)

		# query each robot episode
		query_ep_fols = groups_to_ep_fols.get(query_group, [])
		for ep_count, ep_fol in enumerate(query_ep_fols):
			output_path = f"{ep_fol}/{output_stub}"
			if os.path.exists(output_path):
				myprint(f'[cross_domain] skipping episode {ep_fol} (already processed) [ep {ep_count}/{len(query_ep_fols)}]')
				continue
			try:
				this_episode_embeddings = _load_episode_embeddings(ep_fol, embedding_type)
			except Exception as e:
				myprint(f'[cross_domain] skipping episode {ep_fol} due to error loading query embeddings: {e}')
				continue
			this_num_query = len(this_episode_embeddings)
			robot_ep_idx = fols_to_ep_idxs[ep_fol]
			myprint(f'[cross_domain] querying for {ep_fol} with {this_num_query} steps [ep {ep_count}/{len(query_ep_fols)}]')

			# search
			topk_distances, topk_indices = knn_index.search(this_episode_embeddings, 2 * knn_k)
			try:
				topk_indices = np.array([[idx for idx in indices if idx != -1][:knn_k] for indices in topk_indices])
			except:
				print(f'---------------------------------------------------Too many -1s from topk_indices (cross_domain) ----------------------------------------------------')
				temp_topk_indices = [[idx for idx in indices if idx != -1][:knn_k] for indices in topk_indices]
				print(f'after -1s, min len: {min([len(indices) for indices in temp_topk_indices])}, max len {max([len(indices) for indices in temp_topk_indices])}')
				print(f'-------------------------------------------------------------------------------------------------------------------------------------------')
				print(f'Leaving some -1s in topk_indices and continuing')
				topk_indices = np.array([row+[-1 for _ in range(knn_k-len(row))] for row in temp_topk_indices])
			retrieved_indices = corpus_indices[topk_indices]
			assert retrieved_indices.shape == (this_num_query, knn_k, 2)

			# compute distances relative to the first neighbor (anchor)
			myprint(f'[cross_domain] calculating distances ...')
			all_distances = []
			for ct in range(this_num_query):
				row = retrieved_indices[ct]
				anchor_ep_idx, anchor_step_idx = row[0]
				anchor_embedding = corpus_embeddings_map[anchor_ep_idx][anchor_step_idx]
				distances = [0.0] + [np.linalg.norm(corpus_embeddings_map[e_idx][s_idx] - anchor_embedding) for e_idx, s_idx in row[1:]]
				distances.append(np.linalg.norm(this_episode_embeddings[ct] - anchor_embedding))
				all_distances.append(distances)
			all_distances = np.array(all_distances)
			assert all_distances.shape == (this_num_query, knn_k + 1), f'{all_distances.shape=} {this_num_query=} {knn_k=}'

			# save
			query_indices = np.array([[robot_ep_idx, stp_idx] for stp_idx in range(this_num_query)], dtype=np.int32)
			to_save = {
				'retrieved_indices': retrieved_indices.astype(np.int32),
				'query_indices': query_indices,
				'distances': all_distances,
			}
			np.savez(output_path, **to_save)
			myprint(f'[cross_domain] saved retrieval indices for {ep_fol}')
	myprint(f'[cross_domain] done!')

def create_idx_fol_mapping(ds_name):
	mapping_names = ['groups_to_ep_fols', 'ep_idxs_to_fol', 'fols_to_ep_idxs', 'groups_to_ep_idxs']
	mappings = {temp_name: defaultdict(list) if temp_name == 'groups_to_ep_idxs' else {} for temp_name in mapping_names}
	
	count = 100000 # count starts from 100k because droid has less than 100k episodes
	groups = [f'{ds_name}/{dir}' for dir in os.listdir(ds_name) if not (dir.startswith('.git') or dir.endswith('.json') or dir == 'README.md')]
	mappings['groups_to_ep_fols'] = {group: [f'{group}/{fol}' for fol in os.listdir(group)] for group in groups}
	for group, ep_fols in mappings['groups_to_ep_fols'].items():
		for ep_fol in ep_fols:
			mappings['ep_idxs_to_fol'][count] = ep_fol
			mappings['fols_to_ep_idxs'][ep_fol] = count
			mappings['groups_to_ep_idxs'][group].append(count)
			count += 1

	# first delete the files if they exist
	for file_stub in mapping_names:
		if os.path.exists(f'{ds_name}/{file_stub}.json'):
			os.remove(f'{ds_name}/{file_stub}.json')
	# save ep_idxs_to_fol and fols_to_ep_idxs as jsons
	for file_stub in mapping_names:
		with open(f'{ds_name}/{file_stub}.json', 'w') as f:
			json.dump(mappings[file_stub], f, indent=4)

	return mappings

def create_idx_fol_mapping_for_a_single_group(ds_name):
	mapping_names = ['groups_to_ep_fols', 'ep_idxs_to_fol', 'fols_to_ep_idxs', 'groups_to_ep_idxs']
	# if these files exist, skip this group
	for file_stub in mapping_names:
		if os.path.exists(f'{ds_name}/{file_stub}.json'):
			print(f'skipping {ds_name=} because {file_stub}.json exists. If you want to re-run, delete the four json files at {ds_name}/.')
			return None
	
	# the create the mappings
	mappings = {temp_name: defaultdict(list) if temp_name == 'groups_to_ep_idxs' else {} for temp_name in mapping_names}
	
	count = 100000 # count starts from 100k for this single group
	groups = [f'{ds_name}']
	mappings['groups_to_ep_fols'] = {group: [f'{group}/{fol}' for fol in os.listdir(group)] for group in groups}
	for group, ep_fols in mappings['groups_to_ep_fols'].items():
		for ep_fol in ep_fols:
			mappings['ep_idxs_to_fol'][count] = ep_fol
			mappings['fols_to_ep_idxs'][ep_fol] = count
			mappings['groups_to_ep_idxs'][group].append(count)
			count += 1

	# save ep_idxs_to_fol and fols_to_ep_idxs as jsons
	for file_stub in mapping_names:
		with open(f'{ds_name}/{file_stub}.json', 'w') as f:
			json.dump(mappings[file_stub], f, indent=4)

	return mappings

def retrieval_preprocessing(groups_to_ep_idxs, ep_idxs_to_fol, nb_cores_autofaiss, knn_k, embedding_type):
	myprint(f'[retrieval_preprocessing] starting retrieval preprocessing for {embedding_type}')
	
	# init setup
	num_groupings = len(groups_to_ep_idxs)

	# main loop
	for chosen_id_count, (chosen_id, ep_idxs) in enumerate(groups_to_ep_idxs.items()):
		# count
		total_episodes_in_grouping = len(ep_idxs)
		
		# collect all embeddings and indices
		all_embeddings = []
		all_embeddings_map = {}
		all_indices = []
		for ep_count, ep_idx in enumerate(ep_idxs):
			if embedding_type in EMBED_TYPES:
				ep_embeddings = np.load(f"{ep_idxs_to_fol[ep_idx]}/processed_demo.npz")[f"{embedding_type}_embeddings"]
				all_embeddings.append(ep_embeddings)
				all_embeddings_map[ep_idx] = ep_embeddings
			elif embedding_type == "both":
				ep_embeddings = np.concatenate([np.load(f"{ep_idxs_to_fol[ep_idx]}/processed_demo.npz")[f"{item}_embeddings"] for item in EMBED_TYPES], axis=1)
				all_embeddings.append(ep_embeddings)
				all_embeddings_map[ep_idx] = ep_embeddings
			else:
				raise ValueError(f'{embedding_type=} is not in {EMBED_TYPES} and not "both"')
			num_steps = len(ep_embeddings)
			all_indices.extend([[ep_idx, stp_idx] for stp_idx in range(num_steps)])
		all_embeddings = np.concatenate(all_embeddings, axis=0)
		all_indices = np.array(all_indices)
		embedding_dim = all_embeddings.shape[1]
		num_total = len(all_embeddings)
		myprint(f'[retrieval_preprocessing] concatenated all embeddings and indices for {total_episodes_in_grouping} episodes for {chosen_id} [chosen_id count {chosen_id_count}/{num_groupings}]')
		myprint(f'[retrieval_preprocessing] we have {num_total=} {embedding_dim=}')

		# for each episode, retrieve from all other embeddings
		for ep_count, ep_idx in enumerate(ep_idxs):
			if os.path.exists(f"{ep_idxs_to_fol[ep_idx]}/indices_and_distances.npz"):
				myprint(f'[retrieval_preprocessing] skipping episode {ep_idx} [episode count {ep_count}/{total_episodes_in_grouping}]')
				continue

			all_other_episodes_mask = np.array([True if ep_idx_other != ep_idx else False for (ep_idx_other, stp_idx_other) in all_indices])
			num_retrieval = np.sum(all_other_episodes_mask)
			this_episode_mask = np.array([True if ep_idx_other == ep_idx else False for (ep_idx_other, stp_idx_other) in all_indices])
			num_query = np.sum(this_episode_mask)
			assert num_retrieval + num_query == num_total
			print(f'[retrieval_preprocessing] for episode {ep_idx} [episode count {ep_count}/{total_episodes_in_grouping}], we have {num_retrieval=} {num_query=}')

			# retrieve based on closeness in each type of embedding
			all_other_episodes_embeddings = all_embeddings[all_other_episodes_mask]
			all_other_episodes_indices = all_indices[all_other_episodes_mask]
			this_episode_embeddings = all_embeddings[this_episode_mask]
			this_episode_indices = all_indices[this_episode_mask]
			assert all_other_episodes_embeddings.shape == (num_retrieval, embedding_dim) and all_other_episodes_indices.shape == (num_retrieval, 2), f'{all_other_episodes_embeddings.shape=} {all_other_episodes_indices.shape=}, {num_retrieval=} {embedding_dim=}'
			assert this_episode_embeddings.shape == (num_query, embedding_dim) and this_episode_indices.shape == (num_query, 2)
			assert this_episode_indices.dtype == np.int64 and all_other_episodes_indices.dtype == np.int64

			# create index with all_other_episodes_embeddings
			knn_index, knn_index_infos = build_index(embeddings=all_other_episodes_embeddings, # Note: embeddings have to be float to avoid errors in autofaiss / embedding_reader!
                                            save_on_disk=False,
                                            min_nearest_neighbors_to_retrieve=knn_k + 5, # default: 20
                                            max_index_query_time_ms=10, # default: 10
                                            max_index_memory_usage="25G", # default: "16G"
                                            current_memory_available="50G", # default: "32G"
                                            metric_type='l2',
                                            nb_cores=nb_cores_autofaiss, # default: None # "The number of cores to use, by default will use all cores" as seen in https://criteo.github.io/autofaiss/getting_started/quantization.html#the-build-index-command
                                            )

			# do retrieval from index for this_episode_embeddings
			topk_distances, topk_indices = knn_index.search(this_episode_embeddings, 2 * knn_k)

			# remove -1s and crop to knn_k
			try:
				topk_indices = np.array([[idx for idx in indices if idx != -1][:knn_k] for indices in topk_indices])
			except:
				print(f'---------------------------------------------------Too many -1s from topk_indices ----------------------------------------------------')
				temp_topk_indices = [[idx for idx in indices if idx != -1][:knn_k] for indices in topk_indices]
				print(f'after -1s, min len: {min([len(indices) for indices in temp_topk_indices])}, max len {max([len(indices) for indices in temp_topk_indices])}')
				print(f'-------------------------------------------------------------------------------------------------------------------------------------------')
				print(f'Leaving some -1s in topk_indices and continuing')
				topk_indices = np.array([row+[-1 for _ in range(knn_k-len(row))] for row in temp_topk_indices])
			
			# convert topk_indices to ep_idxs and stp_idxs
			retrieved_indices = all_other_episodes_indices[topk_indices]
			assert retrieved_indices.shape == (num_query, knn_k, 2) and retrieved_indices.dtype == np.int64

			# convert to int32
			retrieved_indices = retrieved_indices.astype(np.int32)
			this_episode_indices = this_episode_indices.astype(np.int32)

			# calculate distances between every embedding of retrieved_indices/query_indices and the first retrieved embedding
			myprint(f'[retrieval_preprocessing] calculating distances ...')
			all_distances = []
			for ct in range(num_query):
				retrieved_indices_row = retrieved_indices[ct]
				temp_first_embedding = all_embeddings_map[retrieved_indices_row[0][0]][retrieved_indices_row[0][1]]
				query_ep_idx, query_stp_idx = this_episode_indices[ct]
				assert query_ep_idx == ep_idx and query_stp_idx == ct
				distances = [0.0] + [np.linalg.norm(all_embeddings_map[e_idx][s_idx] - temp_first_embedding) for e_idx, s_idx in retrieved_indices_row[1:]]
				distances.append(np.linalg.norm(all_embeddings_map[query_ep_idx][query_stp_idx] - temp_first_embedding))
				all_distances.append(distances)
			all_distances = np.array(all_distances)
			assert all_distances.shape == (num_query, knn_k + 1), f'{all_distances.shape=} {num_query=} {knn_k=}'

			# save the retrieved indices and this_episode_indices
			np.savez(f"{ep_idxs_to_fol[ep_idx]}/indices_and_distances.npz", 
						retrieved_indices=retrieved_indices, 
						query_indices=this_episode_indices,
						distances=all_distances)
			myprint(f'[retrieval_preprocessing] finished and saved retrieval indices for episode {ep_idx} [episode count {ep_count}/{total_episodes_in_grouping}]')
		myprint(f'[retrieval_preprocessing] finished for {chosen_id} [chosen_id count {chosen_id_count}/{num_groupings}]')
	myprint(f'[retrieval_preprocessing] done for {embedding_type=}!')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--nb_cores_autofaiss", type=int, default=8)
	parser.add_argument("--knn_k", type=int, default=100, help="number of nearest neighbors to retrieve")
	parser.add_argument("--embedding_type", type=str, default="top_image", choices=EMBED_TYPES + ["both"])
	parser.add_argument("--folder_name", type=str, default="collected_demos_training")
	parser.add_argument("--cross_domain", action="store_true", help="Enable cross-domain retrieval: query from *_robot, retrieve from *_human")
	parser.add_argument("--query_suffix", type=str, default="_robot")
	parser.add_argument("--corpus_suffix", type=str, default="_human")
	parser.add_argument("--output_stub", type=str, default="indices_and_distances.npz")
	args = parser.parse_args()

	if args.folder_name == "collected_demos_training" or args.folder_name == "ricl_human_demo":
		# setup
		ds_name = args.folder_name
		mappings = create_idx_fol_mapping(ds_name)

		# retrieval preprocessing
		if args.cross_domain:
			retrieval_preprocessing_cross_domain(ds_name=ds_name,
											  mappings=mappings,
											  nb_cores_autofaiss=args.nb_cores_autofaiss,
											  knn_k=args.knn_k,
											  embedding_type=args.embedding_type,
											  query_suffix=args.query_suffix,
											  corpus_suffix=args.corpus_suffix,
											  output_stub=args.output_stub)
		else:
			retrieval_preprocessing(groups_to_ep_idxs=mappings['groups_to_ep_idxs'],
									ep_idxs_to_fol=mappings['ep_idxs_to_fol'],
									nb_cores_autofaiss=args.nb_cores_autofaiss, 
									knn_k=args.knn_k,
									embedding_type=args.embedding_type)
		print(f'done!')
	elif args.folder_name == "collected_demos":
		all_groups_in_folder = [f"{args.folder_name}/{fol}" for fol in os.listdir(args.folder_name) if os.path.isdir(f"{args.folder_name}/{fol}")]
		for fol_count, fol_name in enumerate(all_groups_in_folder):
			# setup
			ds_name = fol_name
			mappings = create_idx_fol_mapping_for_a_single_group(ds_name)
			if mappings is None: # skip this group if the files already exist
				continue

			# retrieval preprocessing
			retrieval_preprocessing(groups_to_ep_idxs=mappings['groups_to_ep_idxs'],
									ep_idxs_to_fol=mappings['ep_idxs_to_fol'],
									nb_cores_autofaiss=args.nb_cores_autofaiss, 
									knn_k=args.knn_k,
									embedding_type=args.embedding_type)
			print(f'done for {ds_name=}! [fol count {fol_count}/{len(all_groups_in_folder)}]')
		print(f'done!')
