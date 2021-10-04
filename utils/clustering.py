from sklearn.cluster import KMeans


def kmean(embeds, n_clusters, random_state: int = 0):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    kmeans.fit(embeds)
    kmean_pred = kmeans.predict(embeds)

    labels = kmean_pred.tolist()

    return labels


def assign_by_cluster(track_list, labels, num_clusters):
    clusters = [[] for _ in range(num_clusters)]

    for i in range(len(labels)):
        clusters[labels[i]].append(track_list[i][1:].numpy())

    return clusters
