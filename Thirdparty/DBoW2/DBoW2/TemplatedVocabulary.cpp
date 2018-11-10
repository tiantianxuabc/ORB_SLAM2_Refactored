#include "TemplatedVocabulary.h"

namespace DBoW2 {

template class TemplatedVocabulary<FORB::TDescriptor, FORB>;

struct GetInt
{
	GetInt(char* buffer) { str = strtok(buffer, " "); }

	inline int operator()()
	{
		const int i = atoi(str);
		str = strtok(NULL, " ");
		return i;
	}

	char* str;
};

template<>
bool TemplatedVocabulary<FORB::TDescriptor, FORB>::loadFromTextFile(const std::string &filename)
{
	FILE* fp = fopen(filename.c_str(), "r");
	if (!fp)
		return false;

	const int MAX_COUNT = 256;
	char buffer[MAX_COUNT];
	if (!fgets(buffer, MAX_COUNT, fp))
		return false;

	int n1, n2;
	sscanf(buffer, "%d %d %d %d", &m_k, &m_L, &n1, &n2);

	if (m_k < 0 || m_k>20 || m_L < 1 || m_L>10 || n1 < 0 || n1>5 || n2 < 0 || n2>3)
	{
		std::cerr << "Vocabulary loading failure: This is not a correct text file!" << endl;
		return false;
	}

	m_scoring = (ScoringType)n1;
	m_weighting = (WeightingType)n2;
	createScoringObject();

	// nodes
	int expected_nodes =
		(int)((pow((double)m_k, (double)m_L + 1) - 1) / (m_k - 1));
	m_nodes.reserve(expected_nodes);

	m_words.reserve(pow((double)m_k, (double)m_L + 1));

	m_nodes.resize(1);
	m_nodes[0].id = 0;

	while (fgets(buffer, MAX_COUNT, fp))
	{
		int nid = m_nodes.size();
		m_nodes.resize(m_nodes.size() + 1);
		m_nodes[nid].id = nid;
		m_nodes[nid].descriptor.create(cv::Size(FORB::L, 1), CV_8U);

		GetInt getInt(buffer);
		int pid = getInt();
		int nIsLeaf = getInt();
		for (int i = 0; i < FORB::L; i++)
			m_nodes[nid].descriptor.data[i] = static_cast<uchar>(getInt());
		int weight = getInt();

		m_nodes[nid].parent = pid;
		m_nodes[pid].children.push_back(nid);

		m_nodes[nid].weight = weight;

		if (nIsLeaf > 0)
		{
			int wid = m_words.size();
			m_words.resize(wid + 1);

			m_nodes[nid].word_id = wid;
			m_words[wid] = &m_nodes[nid];
		}
		else
		{
			m_nodes[nid].children.reserve(m_k);
		}
	}

	return true;
}

} // namespace DBoW2
