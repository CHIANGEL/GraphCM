import os
import pprint
import copy

def check_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def save_dict(file_path, file_name, dict):
    check_path(file_path)
    data_path = os.path.join(file_path, file_name)
    file = open(data_path, 'w')
    file.write(str(dict))
    file.close()

def load_dict(file_path, file_name):
    data_path = os.path.join(file_path, file_name) if file_path is not None else file_name
    assert os.path.isfile(data_path), '{} file does not exist.'.format(data_path)
    file = open(data_path, 'r')
    return eval(file.read())
    
def save_list(file_path, file_name, list_data):
    check_path(file_path)
    data_path = os.path.join(file_path, file_name)
    file = open(data_path, 'w')
    file.write(str(list_data))
    file.close()

def load_list(file_path, file_name):
    data_path = os.path.join(file_path, file_name)
    assert os.path.isfile(data_path), '{} file does not exist.'.format(data_path)
    file = open(data_path, 'r')
    return eval(file.read())

def generate_data_per_query(infos_per_query, indices, file_path, file_name):
    check_path(file_path)
    data_path_quid = os.path.join(file_path, '{}_quid.txt'.format(file_name))
    file_quid = open(data_path_quid, 'w')
    query_list, doc_list, vtype_list, cnt_click = [], [], [], 0
    for key in indices:
        interaction_info = infos_per_query[key]
        sid = interaction_info['sid']
        qid = interaction_info['qid']
        uids = interaction_info['uids']
        vids = interaction_info['vids']
        clicks = interaction_info['clicks']
        query_list.append(qid)
        doc_list += uids
        vtype_list += vids
        cnt_click += sum(clicks)
        file_quid.write("{}\t{}\t{}\t{}\t{}\n".format(sid, qid, str(uids), str(vids), str(clicks)))
    print('      - {}'.format('total/unique query: {}, {}'.format(len(query_list), len(set(query_list)))))
    print('      - {}'.format('total/unique document: {}, {}'.format(len(doc_list), len(set(doc_list)))))
    print('      - {}'.format('total/unique vtype: {}, {}'.format(len(vtype_list), len(set(vtype_list)))))
    print('      - {}'.format('total clicks: {}'.format(cnt_click)))
    file_quid.close()
    return set(query_list), set(doc_list), set(vtype_list)

def generate_data_per_query_for_human_label(relevance_queries, infos_per_session, indices, file_path, file_name):
    # Match and resort the every 10 docs within a query session
    print('  - {}'.format('Match and resorting every 10 docs with a query session'))
    cnt = 0
    sid_found, qid_found, uid_match = False, False, False
    for idx, info_per_query in enumerate(relevance_queries):
        id = info_per_query['id']
        sid = info_per_query['sid']
        qid = info_per_query['qid']
        uids = info_per_query['uids']
        relevances = info_per_query['relevances']
        sid_found = False
        for s_idx, info_per_session in enumerate(infos_per_session):
            if sid == info_per_session['sid']:
                sid_found = True
                interaction_infos = info_per_session['interactions']
                qid_found = False
                for i_idx, interaction_info in enumerate(interaction_infos):
                    if qid == interaction_info['qid']:
                        qid_found = True
                        session_uids = interaction_info['uids']
                        session_uids_set = set(session_uids)
                        uids_set = set(uids)
                        if session_uids_set == uids_set:
                            uid_match = True
                            uid_rel = {}
                            for r_idx, rel in enumerate(relevances):
                                uid_rel[uids[r_idx]] = rel
                            relevance_queries[idx]['uids'] = interaction_info['uids']
                            relevance_queries[idx]['vids'] = interaction_info['vids']
                            relevance_queries[idx]['clicks'] = interaction_info['clicks']
                            relevance_queries[idx]['relevances'] = [uid_rel[uid] for uid in interaction_info['uids']]
                            break
                        else:
                            uid_match = False
                if sid_found and qid_found and uid_match:
                    assert relevance_queries[idx]['uids'] == infos_per_session[s_idx]['interactions'][i_idx]['uids']
                    assert relevance_queries[idx]['vids'] == infos_per_session[s_idx]['interactions'][i_idx]['vids']
                    assert relevance_queries[idx]['clicks'] == infos_per_session[s_idx]['interactions'][i_idx]['clicks']
                    assert sorted(relevance_queries[idx]['relevances']) == sorted(info_per_query['relevances'])
                    cnt += 1
                    if cnt % 500 == 0:
                        print('    - {}'.format('match {} sessions'.format(cnt)))
                    sid_found, qid_found, uid_match = False, False, False
                    break
                else:
                    print('  - {}'.format('cannot match the {}-th session:'.format(cnt)))
                    print('    - {}'.format('{}: {}'.format('sid', sid)))
                    print('    - {}'.format('{}: {}'.format('qid', qid)))
                    print('    - {}'.format('{}: {}'.format('uids', uids)))
                    assert 0
    assert cnt == 2000

    # Write label files with qid/uid
    check_path(file_path)
    data_path = os.path.join(file_path, '{}_quid.txt'.format(file_name))
    file = open(data_path, 'w')
    print('  - {}'.format('writing into {} with qid/uid'.format(data_path)))
    query_list, doc_list, vtype_list, cnt_click = [], [], [], 0
    for key in indices:
        query_sequence_for_print = []
        prev_document_info_for_print = []
        info_per_query = relevance_queries[key]
        id = info_per_query['id']
        sid = info_per_query['sid']
        qid = info_per_query['qid']
        uids = info_per_query['uids']
        vids = info_per_query['vids']
        clicks = info_per_query['clicks']
        relevances = info_per_query['relevances']
        query_list.append(qid)
        doc_list += uids
        vtype_list += vids
        cnt_click += sum(clicks)
        file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(sid, qid, str(uids), str(vids), str(clicks), str(relevances)))
    print('    - {}'.format('total/unique query: {}, {}'.format(len(query_list), len(set(query_list)))))
    print('    - {}'.format('total/unique document: {}, {}'.format(len(doc_list), len(set(doc_list)))))
    print('    - {}'.format('total/unique vtype: {}, {}'.format(len(vtype_list), len(set(vtype_list)))))
    print('    - {}'.format('total clicks: {}'.format(cnt_click)))
    file.close()

def generate_data_per_session_for_human_label(relevance_queries, infos_per_session, indices, file_path, file_name):
    # NOTE: The following implementation is derived from FUNC generate_data_per_query_for_human_label, 
    #       so that it may be redundant and insufficient.
    # Match and resort the every 10 docs within a query session
    print('  - {}'.format('Match and resorting every 10 docs with a query session'))
    check_path(file_path)
    data_path = os.path.join(file_path, '{}_quid.txt'.format(file_name))
    file = open(data_path, 'w')
    print('  - {}'.format('writing into {} with qid/uid'.format(data_path)))
    cnt = 0
    sid_found, qid_found, uid_match = False, False, False
    for idx, info_per_query in enumerate(relevance_queries):
        id = info_per_query['id']
        sid = info_per_query['sid']
        qid = info_per_query['qid']
        uids = info_per_query['uids']
        relevances = info_per_query['relevances']
        sid_found = False
        for s_idx, info_per_session in enumerate(infos_per_session):
            if sid == info_per_session['sid']:
                sid_found = True
                interaction_infos = info_per_session['interactions']
                qid_found = False
                for i_idx, interaction_info in enumerate(interaction_infos):
                    if qid == interaction_info['qid']:
                        qid_found = True
                        session_uids = interaction_info['uids']
                        session_uids_set = set(session_uids)
                        uids_set = set(uids)
                        if session_uids_set == uids_set:
                            uid_match = True
                            uid_rel = {}
                            for r_idx, rel in enumerate(relevances):
                                uid_rel[uids[r_idx]] = rel
                            relevance_queries[idx]['uids'] = interaction_info['uids']
                            relevance_queries[idx]['vids'] = interaction_info['vids']
                            relevance_queries[idx]['clicks'] = interaction_info['clicks']
                            relevance_queries[idx]['relevances'] = [uid_rel[uid] for uid in interaction_info['uids']]
                            session_print = copy.deepcopy(info_per_session)
                            session_print['interactions'][i_idx]['relevances'] = relevance_queries[idx]['relevances'].copy()
                            break
                        else:
                            uid_match = False
                if sid_found and qid_found and uid_match:
                    assert relevance_queries[idx]['uids'] == infos_per_session[s_idx]['interactions'][i_idx]['uids']
                    assert relevance_queries[idx]['vids'] == infos_per_session[s_idx]['interactions'][i_idx]['vids']
                    assert relevance_queries[idx]['clicks'] == infos_per_session[s_idx]['interactions'][i_idx]['clicks']
                    assert sorted(relevance_queries[idx]['relevances']) == sorted(info_per_query['relevances'])
                    interaction_infos = session_print['interactions']
                    sid = session_print['sid']
                    for i_idx, interaction_info in enumerate(interaction_infos):
                        qid = interaction_info['qid']
                        uids = interaction_info['uids']
                        vids = interaction_info['vids']
                        clicks = interaction_info['clicks']
                        relevances = interaction_info['relevances'] if 'relevances' in interaction_info else [-1] * 10
                        file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(sid, qid, str(uids), str(vids), str(clicks), str(relevances)))
                    cnt += 1
                    if cnt % 500 == 0:
                        print('    - {}'.format('match {} sessions'.format(cnt)))
                    sid_found, qid_found, uid_match = False, False, False
                    break
                else:
                    print('  - {}'.format('cannot match the {}-th session:'.format(cnt)))
                    print('    - {}'.format('{}: {}'.format('sid', sid)))
                    print('    - {}'.format('{}: {}'.format('qid', qid)))
                    print('    - {}'.format('{}: {}'.format('uids', uids)))
                    assert 0
    assert cnt == 2000

def xml_line_removable(xml_line):
    if xml_line.find('<query>') != -1 and xml_line.find('</query>') != -1:
        return 1
    elif xml_line.find('<url>') != -1 and xml_line.find('</url>') != -1:
        return 1
    elif xml_line.find('<title>') != -1 and xml_line.find('</title>') != -1:
        return 1
    elif xml_line.find('<relevance>') != -1 or xml_line.find('</relevance>') != -1:
        return 1
    elif xml_line.find('<TACM>') != -1 and xml_line.find('</TACM>') != -1:
        return 1
    elif xml_line.find('<PSCM>') != -1 and xml_line.find('</PSCM>') != -1:
        return 1
    elif xml_line.find('<THCM>') != -1 and xml_line.find('</THCM>') != -1:
        return 1
    elif xml_line.find('<UBM>') != -1 and xml_line.find('</UBM>') != -1:
        return 1
    elif xml_line.find('<DBN>') != -1 and xml_line.find('</DBN>') != -1:
        return 1
    elif xml_line.find('<POM>') != -1 and xml_line.find('</POM>') != -1:
        return 1
    return 0