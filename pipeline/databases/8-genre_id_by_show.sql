-- Write a script that lists all shows contained in hbtn_0d_tvshows that
-- have at least one genre linked.
select t1.title, t2.genre_id
from tv_shows t1
inner join tv_show_genres t2
on t1.id = t2.show_id
order by t1.title ASC, t2.genre_id