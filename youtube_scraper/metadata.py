import time


def get_video_details(youtube, video_ids):
    """Fetch metadata and statistics for one or more single-video IDs."""
    all_videos = []

    print("\nFetching video metadata and statistics...")

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch)
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})

            video = {
                "id": item["id"],
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "published_at": snippet.get("publishedAt", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "tags": snippet.get("tags", []),
                "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                "duration": content.get("duration", ""),
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "url": f"https://www.youtube.com/watch?v={item['id']}",
                "transcript": None,
                "transcript_language": None,
                "transcript_error": None
            }
            all_videos.append(video)

        print(f"  Processed metadata for {min(i + 50, len(video_ids))}/{len(video_ids)} videos...")
        time.sleep(0.3)

    return all_videos
