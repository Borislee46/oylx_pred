import json
import os
import uuid

import streamlit as st
import streamlit.components.v1 as components


def _get_base_url() -> str:
    dev_config_path = "config/dev_config.json"
    if os.path.exists(dev_config_path):
        with open(dev_config_path, encoding="utf-8") as f:
            dev_config = json.load(f)
        if dev_config.get("DEBUG_MODE", False):
            return "http://localhost:80/"

    with open("config/app_config.json", encoding="utf-8") as f:
        all_configs = json.load(f)
    env = os.environ.get("APP_ENV", "test")
    return all_configs[env]["STREAMLIT_APP_BASE_URL"]


def _generate_card_html(available_buttons: list, base_url: str, trace_id: str) -> str:
    from urllib.parse import quote

    user = st.session_state.get("e2_user_nickname", "unknown")
    email = st.session_state.get("e2_user_email", "")

    cards_html = ""
    trace_param = quote(trace_id or "", safe="")

    for idx, (button_text, path_or_url, is_link) in enumerate(available_buttons):
        if is_link:
            redirect_url = (
                f"{base_url}redirect?url={quote(path_or_url, safe='')}"
                f"&source={quote(button_text, safe='')}"
                f"&user={quote(user, safe='')}"
                f"&email={quote(email, safe='')}"
                f"&trace={trace_param}"
            )
            full_url = redirect_url
        else:
            page_name = path_or_url.replace("pages/", "").replace(".py", "")
            full_url = base_url + page_name
            separator = "&" if "?" in full_url else "?"
            full_url = f"{full_url}{separator}trace={trace_param}"
        data_attrs = (
            f'data-index="{idx}" data-url="{full_url}" data-is-link="{str(is_link).lower()}"'
        )
        cards_html += (
            f'<div class="card-wrapper" {data_attrs}>'
            f'<div class="card"><span class="card-text">{button_text}</span></div>'
            f"</div>"
        )
    return cards_html


def _generate_component_html(available_buttons: list, base_url: str, trace_id: str) -> str:
    num_cards = len(available_buttons)
    cards_html = _generate_card_html(available_buttons, base_url, trace_id)

    card_width = 160 if num_cards > 4 else 180
    card_height = 220 if num_cards > 4 else 240
    linear_gap = 32 if num_cards > 4 else 40

    return f"""
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        .poker-container {{
            position: relative;
            width: 100%;
            height: 420px;
            display: flex;
            justify-content: center;
            align-items: flex-end;
            padding-bottom: 40px;
            overflow: visible;
            perspective: 1200px;
        }}
        
        .cards-wrapper {{
            position: relative;
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: flex-end;
            transform-style: preserve-3d;
        }}
        
        .card-wrapper {{
            position: absolute;
            width: {card_width}px;
            height: {card_height}px;
            cursor: pointer;
            transform-origin: center 350px;
            transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        }}
        
        .card {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(145deg, #ffffff, #f0f4f8);
            border-radius: 16px;
            border: 2px solid #e2e8f0;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
            box-shadow: 
                0 10px 30px -10px rgba(0, 0, 0, 0.15),
                0 5px 15px -5px rgba(0, 0, 0, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.8);
            pointer-events: none;
        }}
        
        .card::before {{
            content: '';
            position: absolute;
            top: 8px;
            left: 8px;
            right: 8px;
            bottom: 8px;
            border: 1px solid rgba(0, 106, 96, 0.1);
            border-radius: 12px;
            pointer-events: none;
            transition: border-color 0.3s ease;
        }}
        
        .card::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border-radius: 16px;
            background: linear-gradient(105deg, transparent 40%, rgba(255, 255, 255, 0.5) 45%, rgba(255, 255, 255, 0.8) 50%, rgba(255, 255, 255, 0.5) 55%, transparent 60%);
            background-size: 250% 100%;
            background-position: 100% 0;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        
        .card-text {{
            font-size: 16px;
            font-weight: 600;
            color: #334155;
            text-align: center;
            padding: 20px;
            line-height: 1.4;
            z-index: 1;
            transition: all 0.3s ease;
        }}
        
        .cards-wrapper.fan .card-wrapper {{
            position: absolute;
            bottom: 0;
        }}
        
        .cards-wrapper.fan .card-wrapper.active .card {{
            box-shadow: 
                0 25px 50px -12px rgba(0, 106, 96, 0.25),
                0 15px 30px -8px rgba(0, 0, 0, 0.15);
            border-color: rgba(0, 106, 96, 0.5);
        }}
        
        .cards-wrapper.fan .card-wrapper.active .card::after {{
            opacity: 1;
            animation: shimmer 0.6s ease-out forwards;
        }}
        
        .cards-wrapper.fan .card-wrapper.active .card-text {{
            color: rgb(0, 106, 96);
        }}
        
        @keyframes shimmer {{
            0% {{ background-position: 100% 0; }}
            100% {{ background-position: -50% 0; }}
        }}
        
        .cards-wrapper.linear {{
            display: flex;
            flex-direction: row;
            justify-content: center;
            align-items: center;
            gap: {linear_gap}px;
            height: 100%;
        }}
        
        .cards-wrapper.linear .card-wrapper {{
            position: relative;
            flex-shrink: 0;
        }}
        
        .cards-wrapper.linear .card-wrapper.hovered .card {{
            box-shadow: 
                0 30px 60px -15px rgba(0, 106, 96, 0.3),
                0 20px 40px -10px rgba(0, 0, 0, 0.15),
                0 0 0 1px rgba(0, 106, 96, 0.1);
            border-color: rgba(0, 106, 96, 0.6);
        }}
        
        .cards-wrapper.linear .card-wrapper.hovered .card::before {{
            border-color: rgba(0, 106, 96, 0.25);
        }}
        
        .cards-wrapper.linear .card-wrapper.hovered .card::after {{
            opacity: 1;
            animation: shimmer 0.8s ease-out forwards;
        }}
        
        .cards-wrapper.linear .card-wrapper.hovered .card-text {{
            color: rgb(0, 106, 96);
            transform: scale(1.02);
        }}
        
        .mode-hint {{
            position: absolute;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 12px;
            color: #94a3b8;
            opacity: 0.7;
            transition: opacity 0.3s ease;
            pointer-events: none;
            white-space: nowrap;
        }}
        
        .poker-container:hover .mode-hint {{
            opacity: 1;
        }}
    </style>
    
    <div class="poker-container">
        <div class="cards-wrapper fan" id="cardsWrapper">
            {cards_html}
        </div>
        <div class="mode-hint" id="modeHint">滚轮切换布局</div>
    </div>
    
    <script>
        (function() {{
            const wrapper = document.getElementById('cardsWrapper');
            const container = document.querySelector('.poker-container');
            const hint = document.getElementById('modeHint');
            const cardWrappers = wrapper.querySelectorAll('.card-wrapper');
            const numCards = {num_cards};
            let currentMode = 'fan';
            let isTransitioning = false;
            let hoverTimeout = null;
            let activeWrapper = null;
            
            function calculateFanAngles() {{
                const maxSpread = Math.min(50, numCards * 12);
                const angleStep = numCards > 1 ? maxSpread / (numCards - 1) : 0;
                const startAngle = -maxSpread / 2;
                const maxDepth = 50;
                const maxBlur = 1.5;
                const horizontalSpacing = 45;
                const centerIndex = (numCards - 1) / 2;
                
                cardWrappers.forEach((cw, index) => {{
                    const card = cw.querySelector('.card');
                    const angle = numCards > 1 ? startAngle + (angleStep * index) : 0;
                    const normalizedPos = numCards > 1 ? index / (numCards - 1) : 1;
                    const depth = maxDepth * normalizedPos;
                    const blur = maxBlur * (1 - normalizedPos);
                    const brightness = 0.88 + 0.12 * normalizedPos;
                    const zIndex = index + 1;
                    const offsetX = (index - centerIndex) * horizontalSpacing;
                    
                    if (!cw.classList.contains('active')) {{
                        cw.style.transform = `translateX(${{offsetX}}px) rotate(${{angle}}deg) translateZ(${{depth}}px)`;
                        cw.style.zIndex = zIndex;
                        card.style.filter = `brightness(${{brightness}}) blur(${{blur}}px)`;
                    }}
                    cw.dataset.baseAngle = angle;
                    cw.dataset.baseDepth = depth;
                    cw.dataset.baseZIndex = zIndex;
                    cw.dataset.baseBlur = blur;
                    cw.dataset.baseBrightness = brightness;
                    cw.dataset.baseOffsetX = offsetX;
                }});
            }}
            
            function setActiveWrapper(cw) {{
                if (activeWrapper === cw) return;
                if (activeWrapper) {{
                    const prevCard = activeWrapper.querySelector('.card');
                    activeWrapper.classList.remove('active');
                    activeWrapper.style.transform = `translateX(${{activeWrapper.dataset.baseOffsetX}}px) rotate(${{activeWrapper.dataset.baseAngle}}deg) translateZ(${{activeWrapper.dataset.baseDepth}}px)`;
                    activeWrapper.style.zIndex = activeWrapper.dataset.baseZIndex;
                    prevCard.style.filter = `brightness(${{activeWrapper.dataset.baseBrightness}}) blur(${{activeWrapper.dataset.baseBlur}}px)`;
                }}
                activeWrapper = cw;
                if (cw) {{
                    const card = cw.querySelector('.card');
                    cw.classList.add('active');
                    cw.style.zIndex = 100;
                    const angle = parseFloat(cw.dataset.baseAngle);
                    cw.style.transform = `translateX(${{cw.dataset.baseOffsetX}}px) rotate(${{angle}}deg) translateY(-55px) translateZ(80px) scale(1.05)`;
                    card.style.filter = 'brightness(1) blur(0px)';
                }}
            }}
            
            function switchToFan() {{
                if (currentMode === 'fan' || isTransitioning) return;
                isTransitioning = true;
                
                cardWrappers.forEach((cw, index) => {{
                    cw.style.transition = `all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) ${{index * 40}}ms`;
                    cw.classList.remove('hovered');
                    const card = cw.querySelector('.card');
                    card.style.transform = '';
                }});
                
                wrapper.classList.remove('linear');
                wrapper.classList.add('fan');
                currentMode = 'fan';
                hint.textContent = '滚轮切换布局';
                
                setTimeout(() => {{
                    calculateFanAngles();
                    setTimeout(() => {{
                        cardWrappers.forEach((cw) => {{
                            cw.style.transition = 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)';
                        }});
                        isTransitioning = false;
                    }}, 300 + numCards * 40);
                }}, 50);
            }}
            
            function switchToLinear() {{
                if (currentMode === 'linear' || isTransitioning) return;
                isTransitioning = true;
                setActiveWrapper(null);
                
                const centerIndex = (numCards - 1) / 2;
                cardWrappers.forEach((cw, index) => {{
                    const card = cw.querySelector('.card');
                    const distFromCenter = Math.abs(index - centerIndex);
                    const delay = distFromCenter * 50;
                    cw.style.transition = `all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) ${{delay}}ms`;
                    cw.style.transform = 'translateX(0) rotate(0deg) translateZ(0)';
                    cw.style.zIndex = 1;
                    card.style.filter = 'brightness(1) blur(0px)';
                    card.style.transform = '';
                }});
                
                setTimeout(() => {{
                    wrapper.classList.remove('fan');
                    wrapper.classList.add('linear');
                    currentMode = 'linear';
                    hint.textContent = '滚轮切换布局';
                    
                    setTimeout(() => {{
                        cardWrappers.forEach((cw) => {{
                            cw.style.transition = 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)';
                        }});
                        isTransitioning = false;
                    }}, 100);
                }}, 150 + Math.floor(centerIndex) * 50);
            }}
            
            function handleTilt(cw, e) {{
                if (currentMode !== 'linear' || isTransitioning) return;
                
                const card = cw.querySelector('.card');
                const rect = cw.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                
                const rotateX = (y - centerY) / centerY * -10;
                const rotateY = (x - centerX) / centerX * 10;
                
                card.style.transform = `perspective(600px) rotateX(${{rotateX}}deg) rotateY(${{rotateY}}deg) translateY(-12px) scale(1.04)`;
            }}
            
            function resetTilt(cw) {{
                if (currentMode !== 'linear') return;
                const card = cw.querySelector('.card');
                card.style.transform = '';
            }}
            
            calculateFanAngles();
            
            let wheelTimeout = null;
            
            function handleWheel(e) {{
                if (wheelTimeout) return;
                
                if (Math.abs(e.deltaY) < 5) return;
                
                if (e.deltaY > 0 && currentMode === 'fan') {{
                    wheelTimeout = setTimeout(() => {{
                        wheelTimeout = null;
                    }}, 300);
                    switchToLinear();
                }} else if (e.deltaY < 0 && currentMode === 'linear') {{
                    wheelTimeout = setTimeout(() => {{
                        wheelTimeout = null;
                    }}, 300);
                    switchToFan();
                }}
            }}

            window.addEventListener('wheel', handleWheel, {{ passive: true }});

            try {{
                if (window.parent) {{
                    window.parent.addEventListener('wheel', handleWheel, {{ passive: true }});
                    
                    window.addEventListener('unload', () => {{
                        window.parent.removeEventListener('wheel', handleWheel);
                    }});
                }}
            }} catch (e) {{
                console.warn('Cannot bind wheel event to parent window:', e);
            }}
            
            cardWrappers.forEach((cw) => {{
                cw.addEventListener('mouseenter', function() {{
                    if (currentMode === 'fan') {{
                        if (hoverTimeout) clearTimeout(hoverTimeout);
                        hoverTimeout = setTimeout(() => {{
                            setActiveWrapper(this);
                        }}, 80);
                    }} else if (currentMode === 'linear') {{
                        this.classList.add('hovered');
                    }}
                }});
                
                cw.addEventListener('mousemove', function(e) {{
                    if (currentMode === 'linear') {{
                        handleTilt(this, e);
                    }}
                }});
                
                cw.addEventListener('mouseleave', function() {{
                    if (currentMode === 'fan') {{
                        if (hoverTimeout) clearTimeout(hoverTimeout);
                    }} else if (currentMode === 'linear') {{
                        this.classList.remove('hovered');
                        resetTilt(this);
                    }}
                }});
                
                cw.addEventListener('click', function() {{
                    const url = this.getAttribute('data-url');
                    window.open(url, '_blank');
                }});
            }});
            
            wrapper.addEventListener('mouseleave', function() {{
                if (currentMode !== 'fan') return;
                if (hoverTimeout) clearTimeout(hoverTimeout);
                hoverTimeout = setTimeout(() => {{
                    setActiveWrapper(null);
                }}, 150);
            }});
        }})();
    </script>
    """


def render_buttons_grid(available_buttons: list) -> None:
    if not available_buttons:
        return

    base_url = _get_base_url()
    if "session_trace_uuid" not in st.session_state:
        st.session_state.session_trace_uuid = uuid.uuid4().hex
    trace_id = st.session_state.session_trace_uuid

    component_html = _generate_component_html(available_buttons, base_url, trace_id)

    height = 450 if len(available_buttons) <= 4 else 480

    components.html(component_html, height=height, scrolling=False)
