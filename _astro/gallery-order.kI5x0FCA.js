function o(n,a="desc"){return[...n].sort((e,t)=>{if(!e.takenAt)return t.takenAt?1:0;if(!t.takenAt)return-1;const r=e.takenAt.localeCompare(t.takenAt);return a==="asc"?r:-r})}export{o};
