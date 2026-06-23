(function(){
  const toggle=document.querySelector('.nav-toggle');
  const nav=document.querySelector('.site-nav');
  if(toggle&&nav){toggle.addEventListener('click',()=>{const open=nav.classList.toggle('open');toggle.setAttribute('aria-expanded',String(open));});nav.querySelectorAll('a').forEach(a=>a.addEventListener('click',()=>{nav.classList.remove('open');toggle.setAttribute('aria-expanded','false');}));}
  const button=document.querySelector('[data-copy-target]');
  if(button){button.addEventListener('click',async()=>{const target=document.getElementById(button.dataset.copyTarget);if(!target)return;try{await navigator.clipboard.writeText(target.innerText);button.textContent='Copied';setTimeout(()=>button.textContent='Copy',1400);}catch(e){button.textContent='Select text';}});}
})();
