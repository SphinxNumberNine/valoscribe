"""
VLR.gg match page scraper.

Extracts match metadata, VOD URLs, player information, and starting sides
from VLR.gg match pages for Valorant esports analysis.
"""

from __future__ import annotations
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from valoscribe.utils.logger import get_logger

log = get_logger(__name__)


class VLRScraper:
    """
    Scraper for VLR.gg match pages.

    Extracts per-map metadata including:
    - VOD URLs (YouTube links)
    - Player names, teams, and agents
    - Starting sides (attack/defense)
    """

    def __init__(self, timeout: int = 10):
        """
        Initialize VLR scraper.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_match(self, match_url: str) -> dict:
        """
        Scrape a VLR.gg match page for all maps.

        Expected URL format:
            https://www.vlr.gg/<match_id>/<match_slug>/?game=all&tab=overview

        Args:
            match_url: VLR.gg match URL with game=all parameter

        Returns:
            Dictionary with structure:
            {
                "match_url": str,
                "teams": [str, str],  # [team1_name, team2_name]
                "maps": [
                    {
                        "map_number": int,
                        "map_name": str,
                        "vod_url": str,
                        "starting_sides": {"team1": "attack"|"defense", "team2": "attack"|"defense"},
                        "players": [
                            {"name": str, "team": str, "agent": str},
                            ...  # 10 players total
                        ]
                    },
                    ...
                ]
            }
        """
        log.info(f"Scraping match from {match_url}")

        # Ensure URL has game=all parameter
        if '?game=all' not in match_url and '&game=all' not in match_url:
            log.warning("URL doesn't have game=all parameter, adding it")
            separator = '&' if '?' in match_url else '?'
            match_url = f"{match_url}{separator}game=all&tab=overview"

        # Fetch the main match page with game=all
        response = self.session.get(match_url, timeout=self.timeout)
        response.raise_for_status()

        return self.parse_match_html(response.text, match_url)

    def parse_match_html(self, html: str, match_url: str) -> dict:
        """Parse a saved or fetched page using the same production extraction path."""
        soup = BeautifulSoup(html, 'html.parser')

        # Extract team names and logos
        teams, team_logos = self._extract_team_names(soup)
        log.info(f"Teams: {teams}")
        log.debug(f"Team logos: {team_logos}")

        # Extract VOD URLs for each map
        vod_urls = self._extract_vod_urls(soup)
        log.info(f"Found {len(vod_urls)} map VOD URLs")

        # Extract map data from navigation
        map_data = self._extract_map_navigation(soup)
        log.info(f"Found {len(map_data)} maps")

        # Scrape all maps from the same page
        # Note: All map data is in the HTML in separate vlr-rounds containers
        # We already have the soup object, so we can extract all maps at once
        maps = []
        for i, (game_id, map_name) in enumerate(map_data):
            map_number = i + 1
            vod_url = vod_urls[i] if i < len(vod_urls) else None

            log.info(f"Extracting map {map_number}: {map_name} (game_id: {game_id})")

            map_result = self._extract_map_data(
                soup=soup,
                map_number=map_number,
                map_name=map_name,
                vod_url=vod_url,
                teams=teams,
                team_logos=team_logos,
                container_index=i,  # Pass the container index
            )

            if map_result:
                maps.append(map_result)

        return {
            "match_url": match_url,
            "teams": teams,
            "maps": maps,
        }

    def _extract_team_names(self, soup: BeautifulSoup) -> tuple[list[str], list[str]]:
        """
        Extract team names and logo URLs from match page header.

        Returns tuple of ([team1, team2], [logo_url1, logo_url2])
        Logo URLs are used to match teams to the round timeline.
        """
        teams = []
        logos = []

        # Look in the match header for team names and logos
        header = soup.find('div', class_='match-header')
        if header:
            # Find team-name divs within team links
            team_links = header.find_all('a', class_='match-header-link')
            for link in team_links[:2]:
                team_name_div = link.find('div', class_='wf-title-med')
                if team_name_div:
                    teams.append(team_name_div.get_text(strip=True))

                # Extract logo URL for matching
                logo_img = link.find('img')
                if logo_img and logo_img.get('src'):
                    logos.append(logo_img['src'])
                else:
                    logos.append(None)

        # Fallback: try to find team names in game header
        if len(teams) < 2:
            game_header = soup.find('div', class_='vm-stats-game-header')
            if game_header:
                team_divs = game_header.find_all('div', class_='team-name')
                teams = [team.get_text(strip=True) for team in team_divs[:2]]
                logos = [None, None]  # No logos in fallback

        if len(teams) != 2:
            log.warning(f"Expected 2 teams, found {len(teams)}, using placeholders")
            teams = ["Team1", "Team2"]
            logos = [None, None]

        return teams, logos

    def _extract_vod_urls(self, soup: BeautifulSoup) -> list[str]:
        """
        Extract individual map VOD URLs (YouTube links).

        Returns list of YouTube URLs for each map.
        """
        vod_urls = []

        # Current VLR embeds use icon-only links; labels live in their parent.
        embeds = soup.select('.sm-vod[data-site="youtube"]')
        if embeds:
            for embed in embeds:
                link = embed.select_one('a[href]')
                if link and urlparse(link['href']).hostname in {
                    'youtu.be', 'youtube.com', 'www.youtube.com'
                }:
                    vod_urls.append(link['href'])
            return vod_urls

        # Find the match streams/VODs section
        streams_section = soup.find('div', class_='match-streams-bets-container')

        if streams_section:
            # Get all YouTube links
            links = streams_section.find_all('a', href=True)

            for link in links:
                href = link['href']
                link_text = link.get_text(strip=True).lower()

                # Look for "Map 1", "Map 2", etc.
                if 'map ' in link_text and link_text != 'map':
                    if 'youtube.com' in href or 'youtu.be' in href:
                        vod_urls.append(href)

        return vod_urls

    def _extract_map_navigation(
        self, soup: BeautifulSoup
    ) -> list[tuple[str, str]]:
        """
        Extract map game IDs and names from the navigation section.

        Returns list of (game_id, map_name) tuples.
        """
        map_data = []

        # Find the map navigation
        nav = soup.find('div', class_='vm-stats-gamesnav')

        if nav:
            # Find all map selector items
            map_items = nav.find_all(class_='vm-stats-gamesnav-item')

            for item in map_items:
                game_id = item.get('data-game-id')

                # Skip "All Maps" item
                if game_id == 'all':
                    continue

                # Extract map name from text content
                map_text = item.get_text(strip=True)

                # Map text is like "1Corrode" or "2Lotus" - extract just the map name
                # Remove leading digit
                map_name = ''.join([c for c in map_text if not c.isdigit()]).strip()

                if game_id and map_name:
                    map_data.append((game_id, map_name))
                    log.debug(f"Found map: {map_name} (game_id: {game_id})")

        return map_data

    def _extract_map_data(
        self,
        soup: BeautifulSoup,
        map_number: int,
        map_name: str,
        vod_url: Optional[str],
        teams: list[str],
        team_logos: list[str],
        container_index: int,
    ) -> Optional[dict]:
        """
        Extract map data from the already-fetched page.

        The server returns all maps' data in the HTML. This method selects
        the correct vlr-rounds container by index.

        Args:
            soup: BeautifulSoup object of the game=all page
            map_number: Map number (1-5)
            map_name: Name of the map (e.g., "Ascent", "Bind")
            vod_url: YouTube VOD URL for this map
            teams: List of team names [team1, team2]
            container_index: Index of the vlr-rounds container (0-4)

        Returns:
            Dictionary with map data or None if extraction failed
        """
        try:
            # Current VLR statistics are div-based rather than HTML tables.
            modern_games = [game for game in soup.select('.vm-stats-game')
                            if game.get('data-game-id') != 'all']
            if modern_games and modern_games[container_index].select('.ovw-table'):
                game = modern_games[container_index]
                team_data = []
                tables = game.select('.ovw-table')
                first_round = game.select_one('.vlr-rounds-row-col[title]')
                squares = first_round.select('.rnd-sq') if first_round else []
                winner = next((i for i, sq in enumerate(squares)
                               if 'mod-win' in sq.get('class', [])), None)
                if winner is None or len(tables) != 2:
                    raise ValueError('Missing round sides or team statistics')
                winner_attacks = 'mod-t' in squares[winner].get('class', [])
                for index, (team, table) in enumerate(zip(teams, tables)):
                    players = []
                    for row in table.select('.ovw-row'):
                        name = row.select_one('.ovw-player-name')
                        agent = row.select_one('.ovw-agents img[title]')
                        if name and agent:
                            players.append({'name': name.get_text(strip=True),
                                            'agent': agent['title'].lower()})
                    if len(players) != 5:
                        raise ValueError(f'Expected five players for {team}, got {len(players)}')
                    attacks = winner_attacks if index == winner else not winner_attacks
                    team_data.append({'name': team,
                                      'starting_side': 'attack' if attacks else 'defense',
                                      'players': players})
                return {'map_number': map_number, 'map_name': map_name,
                        'vod_url': vod_url, 'teams': team_data}

            # Extract players from both team tables for this specific map
            all_players = self._extract_players(soup, teams, container_index)

            if len(all_players) != 10:
                log.warning(
                    f"Expected 10 players for map {map_number}, found {len(all_players)}"
                )

            # Extract starting sides from the specific container
            starting_sides = self._extract_starting_sides(soup, teams, team_logos, container_index)

            # Group players by team and combine with starting sides
            team_data = []
            for i, team_name in enumerate(teams):
                # Filter players for this team
                team_players = [
                    {"name": p["name"], "agent": p["agent"]}
                    for p in all_players
                    if p["team"] == team_name
                ]

                # Get starting side for this team (team1 or team2)
                side_key = f"team{i + 1}"
                starting_side = starting_sides.get(side_key, "attack")

                team_data.append({
                    "name": team_name,
                    "starting_side": starting_side,
                    "players": team_players,
                })

            return {
                "map_number": map_number,
                "map_name": map_name,
                "vod_url": vod_url,
                "teams": team_data,
            }

        except Exception as e:
            log.error(f"Failed to extract map {map_number} data: {e}")
            return None

    def _extract_players(self, soup: BeautifulSoup, teams: list[str], container_index: int) -> list[dict]:
        """
        Extract player information from team stat tables for a specific map.

        Args:
            soup: BeautifulSoup object of the page
            teams: List of team names [team1, team2]
            container_index: Index of the map (0-4) to get the correct stat tables

        Returns list of player dicts with name, team, agent.
        """
        players = []

        # Find all stat tables
        tables = soup.find_all('table', class_='wf-table-inset')

        # First 2 tables are overall match stats, then each map has 2 tables (one per team)
        # Map 0 (container_index=0) uses tables 2-3
        # Map 1 (container_index=1) uses tables 4-5
        # Map 2 (container_index=2) uses tables 6-7, etc.
        table_start_idx = (container_index + 1) * 2
        table_end_idx = table_start_idx + 2
        map_tables = tables[table_start_idx:table_end_idx]

        if len(map_tables) < 2:
            log.warning(
                f"Expected 2 stat tables for map {container_index}, found {len(map_tables)}"
            )

        # Should have 2 tables (one per team)
        for team_idx, table in enumerate(map_tables):
            team_name = teams[team_idx] if team_idx < len(teams) else f"Team{team_idx + 1}"

            # Find all player rows
            rows = table.find_all('tr')

            for row in rows:
                # Skip header rows
                if row.find('th'):
                    continue

                cells = row.find_all('td')
                if len(cells) < 2:
                    continue

                # First cell: player info (contains nested divs)
                player_cell = cells[0]

                # Player name is in first div inside the anchor tag
                player_link = player_cell.find('a')
                if player_link:
                    # Find the div with font-weight: 700 (player name)
                    name_div = player_link.find('div', style=lambda s: s and 'font-weight: 700' in s)
                    if name_div:
                        player_name = name_div.get_text(strip=True)
                    else:
                        # Fallback: get first div text
                        first_div = player_link.find('div')
                        player_name = first_div.get_text(strip=True) if first_div else "Unknown"
                else:
                    player_name = player_cell.get_text(strip=True)

                # Second cell: agent icon with title attribute
                agent_cell = cells[1]
                agent_img = agent_cell.find('img')

                if agent_img and agent_img.get('title'):
                    agent_name = agent_img['title'].lower()

                    players.append({
                        "name": player_name,
                        "team": team_name,
                        "agent": agent_name,
                    })

                    log.debug(f"Found player: {player_name} ({team_name}) - {agent_name}")

        return players

    def _extract_starting_sides(
        self, soup: BeautifulSoup, teams: list[str], team_logos: list[str], container_index: int
    ) -> dict:
        """
        Extract starting sides from the first round in the timeline.

        Finds the first round, determines which team won and on which side,
        and assigns starting sides accordingly.

        Args:
            soup: BeautifulSoup object of map page
            teams: List of team names [team1, team2]
            team_logos: List of team logo URLs [logo1, logo2]
            container_index: Index of the vlr-rounds container to use (0-4)

        Returns:
            Dictionary: {"team1": "attack"|"defense", "team2": "attack"|"defense"}
        """
        # Find ALL round timeline containers
        all_rounds_containers = soup.find_all('div', class_='vlr-rounds')

        if not all_rounds_containers:
            log.warning("Could not find any vlr-rounds containers, using default starting sides")
            return {"team1": "attack", "team2": "defense"}

        if container_index >= len(all_rounds_containers):
            log.warning(
                f"Container index {container_index} out of range "
                f"(found {len(all_rounds_containers)} containers), using default starting sides"
            )
            return {"team1": "attack", "team2": "defense"}

        # Select the correct container for this map
        rounds_container = all_rounds_containers[container_index]
        log.debug(f"Using vlr-rounds container {container_index}")

        # Get the direct child vlr-rounds-row
        rounds_rows = rounds_container.find_all('div', class_='vlr-rounds-row', recursive=False)

        if not rounds_rows:
            log.warning("Could not find vlr-rounds-row, using default starting sides")
            return {"team1": "attack", "team2": "defense"}

        first_row = rounds_rows[0]

        # Get the first child to find team names order
        children = first_row.find_all(recursive=False)
        if len(children) == 0:
            log.warning("No children in vlr-rounds-row")
            return {"team1": "attack", "team2": "defense"}

        first_child = children[0]

        # Extract team logos from the first child (top and bottom)
        team_divs = first_child.find_all('div', class_='team')
        if len(team_divs) < 2:
            log.warning("Could not find both team divs in first child")
            return {"team1": "attack", "team2": "defense"}

        # Get team logo URLs from the timeline (top and bottom)
        top_team_logo = None
        bottom_team_logo = None
        top_team_name = team_divs[0].get_text(strip=True)
        bottom_team_name = team_divs[1].get_text(strip=True)

        top_img = team_divs[0].find('img')
        if top_img and top_img.get('src'):
            top_team_logo = top_img['src']

        bottom_img = team_divs[1].find('img')
        if bottom_img and bottom_img.get('src'):
            bottom_team_logo = bottom_img['src']

        log.debug(f"Timeline teams - Top: {top_team_name} (logo: {top_team_logo}), Bottom: {bottom_team_name} (logo: {bottom_team_logo})")

        # Find the first round column with a title attribute
        # Look specifically for "Round 1" in the title
        first_round = None
        first_round_title = None
        for child in children[1:]:  # Skip first child (team names)
            title = child.get('title')
            if title:
                # Check if this is specifically Round 1
                if 'Round 1' in title or title.startswith('Round 1'):
                    first_round = child
                    first_round_title = title
                    break
                # Fallback: take first round found if we don't find Round 1
                elif not first_round:
                    first_round = child
                    first_round_title = title

        if not first_round:
            log.warning("Could not find first round with title attribute")
            return {"team1": "attack", "team2": "defense"}

        log.debug(f"Using round with title: {first_round_title}")

        # Within the round, find the rnd-sq with mod-win
        # There are 2 rnd-sq divs (one for each team, top and bottom)
        rnd_sqs = first_round.find_all('div', class_='rnd-sq')

        if len(rnd_sqs) < 2:
            log.warning("Expected 2 rnd-sq elements in first round")
            return {"team1": "attack", "team2": "defense"}

        # Find which one has mod-win
        winning_team_position = None  # 0 = top, 1 = bottom
        winning_side = None  # 'attack' or 'defense'

        for i, rnd_sq in enumerate(rnd_sqs[:2]):
            classes = rnd_sq.get('class', [])

            if 'mod-win' in classes:
                winning_team_position = i

                # Check mod-ct or mod-t
                if 'mod-ct' in classes:
                    winning_side = "defense"
                elif 'mod-t' in classes:
                    winning_side = "attack"

                break

        if winning_team_position is None or winning_side is None:
            log.warning("Could not determine first round winner and side")
            return {"team1": "attack", "team2": "defense"}

        # Determine which team won (by logo URL)
        winning_team_logo = top_team_logo if winning_team_position == 0 else bottom_team_logo
        winning_team_name = top_team_name if winning_team_position == 0 else bottom_team_name
        losing_team_name = bottom_team_name if winning_team_position == 0 else top_team_name

        # Assign sides
        losing_side = "defense" if winning_side == "attack" else "attack"

        log.info(
            f"Round 1 analysis: {winning_team_name} won on {winning_side}, "
            f"{losing_team_name} was on {losing_side}"
        )

        # Map timeline teams to the teams list (team1, team2) using logo URLs
        # team1 and team2 are from _extract_team_names
        log.debug(f"teams[0]={teams[0]}, teams[1]={teams[1]}")
        log.debug(f"team_logos[0]={team_logos[0]}, team_logos[1]={team_logos[1]}")
        log.debug(f"winning_team_logo={winning_team_logo}")

        result = {}
        if winning_team_logo and winning_team_logo == team_logos[0]:
            result = {"team1": winning_side, "team2": losing_side}
            log.info(f"Starting sides: {teams[0]} (team1) = {winning_side}, {teams[1]} (team2) = {losing_side}")
        elif winning_team_logo and winning_team_logo == team_logos[1]:
            result = {"team1": losing_side, "team2": winning_side}
            log.info(f"Starting sides: {teams[0]} (team1) = {losing_side}, {teams[1]} (team2) = {winning_side}")
        else:
            log.warning(
                f"Could not match winning team logo '{winning_team_logo}' to team logos {team_logos}, "
                f"using default sides (may be incorrect!)"
            )
            result = {"team1": "attack", "team2": "defense"}

        return result


def scrape_match(match_url: str, timeout: int = 10) -> dict:
    """
    Convenience function to scrape a VLR.gg match.

    Args:
        match_url: VLR.gg match URL (should have game=all parameter)
        timeout: Request timeout in seconds

    Returns:
        Match metadata dictionary
    """
    scraper = VLRScraper(timeout=timeout)
    return scraper.scrape_match(match_url)
